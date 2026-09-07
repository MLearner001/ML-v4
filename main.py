"""
main.py
Orchestrator utama untuk menjalankan End-to-End Pipeline Prediksi Machining Time CNC
Menggunakan Dual-Layer Bi-LSTM.
"""

import argparse
import os
import sys
import pandas as pd
from batch_gcode_parser import NCParser
from trace_synchronizer import SinuTrainSynchronizer
from dataset_preprocessor import DatasetPreprocessor, SlidingWindowGenerator
from train_bi_lstm import run_training
from inference_pipeline import predict_nc_file

import glob
import random

import tensorflow as tf

# Terapkan Dynamic GPU Memory Growth untuk mencegah TF merampas seluruh VRAM (OOM Protection)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def run_training_pipeline(data_dir: str, out_dir: str, mem_mode: str = "high",
                          resume_model: str = None, resume_scaler: str = None,
                          learning_rate: float = 1e-3, initial_epoch: int = 0,
                          lstm_units: int = 256, epochs: int = 100, batch_size: int = 128):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    output_model = os.path.join(out_dir, "bilstm_feedrate_model.keras")
    scaler_path = os.path.join(out_dir, "scaler.pkl")

    print(f"\n{'='*50}\n[MEMULAI BATCH TRAINING]\nMencari pasangan file G-Code (.mpf) dan Trace (.csv) di folder: {data_dir}\n{'='*50}")

    # Cari semua file G-Code (.mpf atau .nc)
    gcode_files = glob.glob(os.path.join(data_dir, "*.mpf")) + glob.glob(os.path.join(data_dir, "*.nc"))
    if not gcode_files:
        print("[ERROR] Tidak ditemukan file .mpf atau .nc di folder tersebut.")
        sys.exit(1)

    synced_dfs = []

    for gcode_file in gcode_files:
        base_name = os.path.splitext(os.path.basename(gcode_file))[0]

        synced_filename = f"{base_name}_synced.csv"
        synced_output = os.path.join(out_dir, synced_filename)

        print(f"\n--- Memproses Pasangan: {base_name} ---")

        # JIKA SUDAH PERNAH DI-SINKRONISASI (Resume dari Tahap 3)
        if os.path.exists(synced_output):
            print(f"[TAHAP 1 & 2 SKIPPED] Memuat langsung file sinkronisasi dari: {synced_output}")
            df_synced = pd.read_csv(synced_output, low_memory=False)
            synced_dfs.append(df_synced)
            continue

        # Cari pasangan file trace (.csv) jika belum ada cache
        trace_file = os.path.join(data_dir, f"{base_name}.csv")

        if not os.path.exists(trace_file):
            print(f"[WARNING] Melewati {base_name}: Tidak ditemukan file trace pasangannya ({trace_file})")
            continue

        print("[TAHAP 1] Parsing NC & Geometri 3D...")
        parser = NCParser()
        df_parsed = parser.parse_file(gcode_file)

        print("[TAHAP 2] Sinkronisasi Trace SinuTrain...")

        # Deteksi Header & Separator secara otomatis untuk Trace SinuTrain
        # karena sering mengandung metadata di atas dan menggunakan titik koma (;)
        header_idx = 0
        detected_sep = ','

        with open(trace_file, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                # Cari baris yang kemungkinan merupakan header aktual data time series,
                # mengandung time atau variasi f1/s1 atau f1\s1
                if line.startswith('time') or any(k in line for k in ['f1/s1', 'f1\\s1', 'actLineNumber']):
                    # Pastikan kita tidak menangkap baris meta-data (seperti Signal,key,event...)
                    if 'Signal' not in line:
                        header_idx = i
                        if ';' in line:
                            detected_sep = ';'
                        break

        # Membaca trace SinuTrain menggunakan skip-rows dan error_bad_lines/on_bad_lines dinonaktifkan
        try:
            df_trace = pd.read_csv(trace_file, skiprows=header_idx, sep=detected_sep, low_memory=False, on_bad_lines='skip')
        except TypeError:
            # Fallback untuk versi pandas lama
            df_trace = pd.read_csv(trace_file, skiprows=header_idx, sep=detected_sep, low_memory=False, error_bad_lines=False)
        # Bersihkan spasi whitespace di nama kolom
        df_trace.columns = df_trace.columns.str.strip()

        syncer = SinuTrainSynchronizer()
        df_trace_clean = syncer.clean_and_attribute_trace(df_trace, df_parsed['Block_ID'].tolist())
        df_synced = syncer.match_and_calculate_targets(df_parsed, df_trace_clean)

        synced_filename = f"{base_name}_synced.csv"
        synced_output = os.path.join(out_dir, synced_filename)
        df_synced.to_csv(synced_output, index=False)
        print(f"-> Tersinkronisasi ({len(df_synced)} baris), disimpan ke: {synced_output}")

        synced_dfs.append(df_synced)

    if not synced_dfs:
        print("\n[ERROR] Tidak ada satupun pasangan file yang berhasil disinkronisasi.")
        sys.exit(1)

    print(f"\n{'='*50}\n[TAHAP 3] Scaling, Padding & Sequence Windowing (Batch)\n{'='*50}")
    preprocessor = DatasetPreprocessor(window_size=201)

    is_resume_scaler = False
    if resume_scaler and os.path.exists(resume_scaler):
        print(f"[INFO] Memuat resume Scaler State dari: {resume_scaler}")
        preprocessor.load_scalers(resume_scaler)
        is_resume_scaler = True

    # Mengacak (shuffle) daftar file agar representasi variasi gerakan mesin
    # (Drill, 5-Axis, Contour, dsb) tersebar rata di Training dan Validasi.
    # Menggunakan konstanta Seed (42) agar pengacakan selalu sama setiap script di-run.
    random.Random(42).shuffle(synced_dfs)

    split_idx = int(0.8 * len(synced_dfs))
    if split_idx == 0 and len(synced_dfs) > 0:
        split_idx = 1 # Pastikan minimal ada 1 data training

    train_dfs = synced_dfs[:split_idx]
    val_dfs = synced_dfs[split_idx:]

    print(f"Menggunakan {len(train_dfs)} file untuk Training, {len(val_dfs)} file untuk Validasi.")

    if mem_mode == "high":
        print("[INFO] Menggunakan Mode HIGH RAM (Numpy Arrays 3D di Memori).")
        # Preprocessor menerima list dataframe dari berbagai file untuk di-fit scaler dan diubah ke window 3D
        X_train, Y_train = preprocessor.fit_transform_dataset(train_dfs, is_resume=is_resume_scaler)
        X_val, Y_val = [], []
        if val_dfs:
            X_val_list, Y_val_list = [], []
            for df in val_dfs:
                x_p, y_p = preprocessor.transform_file(df, is_training=True)
                X_val_list.append(x_p)
                Y_val_list.append(y_p)
            if X_val_list:
                X_val = np.concatenate(X_val_list, axis=0)
                Y_val = np.concatenate(Y_val_list, axis=0)

        preprocessor.save_scalers(scaler_path)
        print(f"Scaler parameters saved to {scaler_path}")
        print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape if len(X_val) > 0 else 'N/A'}")

        print(f"\n{'='*50}\n[TAHAP 4] Pelatihan Model Dual-Layer Bi-LSTM\n{'='*50}")
        input_shape = (X_train.shape[1], X_train.shape[2])
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        model, history = run_training((X_train, Y_train), (X_val, Y_val) if len(X_val) > 0 else None,
                                      input_shape=input_shape, model_save_path=output_model,
                                      checkpoint_dir=ckpt_dir, resume_model_path=resume_model,
                                      learning_rate=learning_rate, initial_epoch=initial_epoch,
                                      lstm_units=lstm_units, epochs=epochs)
    else:
        print("[INFO] Menggunakan Mode LOW RAM (On-the-fly Generator). Sangat hemat memori!")
        if not is_resume_scaler:
            preprocessor.fit_scalers_only(train_dfs)
        preprocessor.save_scalers(scaler_path)

        train_gen = SlidingWindowGenerator(train_dfs, preprocessor, batch_size=batch_size)
        val_gen = SlidingWindowGenerator(val_dfs, preprocessor, batch_size=batch_size) if val_dfs else None

        input_shape = (preprocessor.window_size, len(preprocessor.feature_cols))

        print(f"\n{'='*50}\n[TAHAP 4] Pelatihan Model Dual-Layer Bi-LSTM\n{'='*50}")
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        model, history = run_training(train_gen, val_gen, input_shape=input_shape,
                                      model_save_path=output_model, checkpoint_dir=ckpt_dir,
                                      resume_model_path=resume_model, learning_rate=learning_rate,
                                      initial_epoch=initial_epoch, lstm_units=lstm_units,
                                      epochs=epochs)

    print(f"Model berhasil dilatih dan disimpan di: {output_model}")
    print("End-to-End Training Pipeline selesai.\n")

def run_inference_pipeline(data_dir: str, out_dir: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_path = os.path.join(out_dir, "bilstm_feedrate_model.keras")
    scaler_path = os.path.join(out_dir, "scaler.pkl")

    print(f"\n{'='*50}\n[TAHAP 5] Standalone Inference (Batch)\n{'='*50}")

    # Cari semua file G-Code (.mpf atau .nc) di folder test
    gcode_files = glob.glob(os.path.join(data_dir, "*.mpf")) + glob.glob(os.path.join(data_dir, "*.nc"))
    if not gcode_files:
        print(f"[ERROR] Tidak ditemukan file .mpf atau .nc di folder: {data_dir}")
        sys.exit(1)

    print(f"Ditemukan {len(gcode_files)} file program untuk diinferensi.")

    for idx, gcode_file in enumerate(gcode_files):
        print(f"\n[{idx+1}/{len(gcode_files)}] Memproses Prediksi: {os.path.basename(gcode_file)}")
        predict_nc_file(gcode_file, model_path=model_path, scaler_path=scaler_path, out_dir=out_dir)

    print("\n[INFO] Seluruh proses inferensi batch selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Digital Twin CNC Bi-LSTM")

    subparsers = parser.add_subparsers(dest="mode", help="Mode eksekusi: train atau infer")

    # Subparser untuk mode TRAINING
    train_parser = subparsers.add_parser("train", help="Jalankan Pipeline Pelatihan End-to-End (Batch)")
    train_parser.add_argument("--data-dir", type=str, required=True, help="Path folder data mentah (isi .mpf dan .csv)")
    train_parser.add_argument("--out-dir", type=str, default="output", help="Path folder hasil pipeline (default: output)")
    train_parser.add_argument("--mem-mode", type=str, choices=["high", "low"], default="high", help="Pilih 'high' untuk RAM besar (cepat) atau 'low' untuk RAM kecil (hemat memory).")
    train_parser.add_argument("--resume-model", type=str, default=None, help="Path ke model lama (.keras) untuk melanjutkan pelatihan (Transfer Learning)")
    train_parser.add_argument("--resume-scaler", type=str, default=None, help="Path ke scaler lama (.pkl) agar distribusi metrik tetap konsisten")
    train_parser.add_argument("--lr", type=float, default=0.0005, help="Initial Learning Rate (contoh: 0.0005)")
    train_parser.add_argument("--initial-epoch", type=int, default=0, help="Mulai resume dari epoch ke berapa (agar progress bar benar)")
    train_parser.add_argument("--lstm-units", type=int, default=128, help="Kapasitas neuron model untuk Fase 2 (default: 128)")
    train_parser.add_argument("--epochs", type=int, default=100, help="Total epochs untuk pelatihan")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Ukuran batch untuk generator (low RAM) / numpy (high RAM)")

    # Subparser untuk mode INFERENCE
    infer_parser = subparsers.add_parser("infer", help="Jalankan Pipeline Prediksi Standalone (Batch)")
    infer_parser.add_argument("--data-dir", type=str, required=True, help="Path folder G-Code baru (.mpf) untuk testing")
    infer_parser.add_argument("--out-dir", type=str, default="output", help="Path folder berisi model/scaler (dan tempat menyimpan hasil prediksi)")

    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.exists(args.data_dir) or not os.path.isdir(args.data_dir):
            print("[ERROR] Pastikan argumen --data-dir adalah folder yang valid.")
            sys.exit(1)
        run_training_pipeline(args.data_dir, args.out_dir, args.mem_mode,
                              args.resume_model, args.resume_scaler, args.lr,
                              args.initial_epoch, args.lstm_units,
                              args.epochs, args.batch_size)

    elif args.mode == "infer":
        if not os.path.exists(args.data_dir) or not os.path.isdir(args.data_dir):
            print("[ERROR] Pastikan argumen --data-dir adalah folder yang valid.")
            sys.exit(1)

        model_path = os.path.join(args.out_dir, "bilstm_feedrate_model.keras")
        scaler_path = os.path.join(args.out_dir, "scaler.pkl")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"[ERROR] Model atau Scaler tidak ditemukan di {args.out_dir}. Lakukan train terlebih dahulu.")
            sys.exit(1)

        run_inference_pipeline(args.data_dir, args.out_dir)

    else:
        parser.print_help()
