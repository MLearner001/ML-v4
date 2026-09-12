"""
inference_pipeline.py
Tahap 5: Eksekusi Inferensi Standalone pada File Program NC Baru (.mpf).
(Updated: V3 tanpa Hard-Clipping dan dengan fix bug inverse transformation)
"""

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from batch_gcode_parser import NCParser
from dataset_preprocessor import DatasetPreprocessor, SlidingWindowGenerator
from typing import Dict

import os

def predict_nc_file(mpf_filepath: str,
                     model_path: str = "bilstm_feedrate_model.keras",
                     scaler_path: str = "scaler.pkl",
                     out_dir: str = ".") -> Dict[str, float]:
    """Memproses file .mpf baru dan menghitung estimasi waktu pemesinan total."""

    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"[INFO] 1. Mem-parsing file NC: {mpf_filepath}")
    parser = NCParser()
    df_parsed = parser.parse_file(mpf_filepath)

    print(f"[INFO] 2. Memuat Scaler & Menyiapkan Tensor Jendela W=301...")
    preprocessor = DatasetPreprocessor(window_size=301)
    preprocessor.load_scalers(scaler_path)

    # Gunakan SlidingWindowGenerator (mem-mode low) agar WSL/RAM tidak OOM terbunuh paksa pada file gcode besar
    infer_generator = SlidingWindowGenerator([df_parsed], preprocessor, batch_size=256, is_training=False)

    print(f"[INFO] 3. Memuat Model Bi-LSTM & Menjalankan Inferensi...")
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    y_pred_scaled = model.predict(infer_generator, verbose=1)

    # 4. Inverse Transform untuk Mendapatkan Waktu Aktual (Detik)
    y_pred_log = preprocessor.target_scaler.inverse_transform(y_pred_scaled)
    raw_predicted_feedrate = np.maximum(1.0, np.expm1(y_pred_log).flatten())

    # --- SAFEGUARD: PHYSICS-INFORMED CLIPPING ---
    # Ambil batas kecepatan komando (G01 F... atau limit G00) dari parser
    cmd_f_limits = df_parsed['Cmd_F'].values

    # 1. Batas Atas: Tebakan AI TIDAK BOLEH melebihi batas yang diperintahkan program
    predicted_feedrate = np.minimum(raw_predicted_feedrate, cmd_f_limits)

    # 2. Batas Bawah: Tebakan AI minimal 1% dari commanded feedrate atau minimal absolut 1.0 (mencegah anomali waktu meledak, namun membiarkan pengereman tajam 5-Axis)
    min_feedrate_limits = np.maximum(0.01 * cmd_f_limits, 1.0)
    predicted_feedrate = np.maximum(predicted_feedrate, min_feedrate_limits)
    # --------------------------------------------

    # 5. Integrasi Kinematika Fisik & Pengecekan Validitas
    # Gunakan Delta_3D untuk pergerakan linier atau Delta_Rot untuk pergerakan putar
    effective_distance = np.where(df_parsed['Delta_3D'] > 1e-4,
                                  df_parsed['Delta_3D'],
                                  df_parsed['Delta_Rot'])
    # Dapatkan jarak efektif (hindari div-by-zero)
    effective_distance = np.maximum(effective_distance, 1e-6)

    # Ambil Theo_Duration dari parser
    theo_durations = df_parsed['Theo_Duration'].values
    is_motion = df_parsed['Is_Motion_Block'].values

    # Waktu eksekusi:
    # - Jika Motion Block (1): Gunakan Jarak / Prediksi Kecepatan Aktual dari AI
    # - Jika Non-Motion (0): Gunakan durasi mutlak dari parser (contoh: Dwell Time G04)
    df_parsed['Predicted_Feedrate_mm_min'] = predicted_feedrate
    block_durations_sec = np.where(
        is_motion == 1,
        (effective_distance / predicted_feedrate) * 60.0,
        theo_durations  # Mengambil nilai absolut dwell time / delay
    )
    df_parsed['Estimated_Duration_Sec'] = block_durations_sec

    total_time_sec = float(np.sum(block_durations_sec))
    total_time_min = total_time_sec / 60.0

    # Simpan hasil analisis profil feedrate ke CSV
    base_name = os.path.basename(mpf_filepath)
    output_filename = base_name.replace(".mpf", "_predicted_profile.csv").replace(".nc", "_predicted_profile.csv")
    output_csv = os.path.join(out_dir, output_filename)
    df_parsed.to_csv(output_csv, index=False)

    print("\n" + "="*50)
    print("HASIL PREDIKSI MACHINING TIME (DIGITAL TWIN PHASE 2 - V3)")
    print("="*50)
    print(f"Total Blok Program     : {len(df_parsed)} baris")
    print(f"Total Estimasi Waktu   : {total_time_sec:.2f} detik ({total_time_min:.2f} menit)")
    print(f"Profil Lengkap Disimpan: {output_csv}")
    print("="*50 + "\n")

    return {
        "total_seconds": total_time_sec,
        "total_minutes": total_time_min,
        "block_count": len(df_parsed)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_nc_file(sys.argv[1])
    else:
        print("[USAGE] python inference_pipeline.py <path_file_nc.mpf>")
