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
from train_bi_lstm import build_bilstm_model
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

    # Filter hanya Zona Cutting (Segment == 2) dengan pengaman .any()
    if 'Segment' in df_parsed.columns and (df_parsed['Segment'] == 2).any():
        total_before = len(df_parsed)
        df_parsed = df_parsed[df_parsed['Segment'] == 2].copy().reset_index(drop=True)
        print(f"[INFO] Filter Segment 2 aktif: Memproses murni Zona Cutting ({len(df_parsed)}/{total_before} baris).")

    print(f"[INFO] 2. Memuat Scaler & Menyiapkan Tensor Jendela W=301...")
    preprocessor = DatasetPreprocessor(window_size=301)
    preprocessor.load_scalers(scaler_path)

    # Gunakan SlidingWindowGenerator (mem-mode low) agar WSL/RAM tidak OOM terbunuh paksa pada file gcode besar
    infer_generator = SlidingWindowGenerator([df_parsed], preprocessor, batch_size=256, is_training=False)

    print(f"[INFO] 3. Memuat Model Bi-LSTM & Menjalankan Inferensi...")
    # Bangun kerangka arsitektur secara lokal (kompatibel dengan Lambda)
    input_shape = (preprocessor.window_size, len(preprocessor.feature_cols))
    model = build_bilstm_model(input_shape=input_shape)

    # Suntikkan bobot hasil training
    model.load_weights(model_path)

    y_pred_scaled = model.predict(infer_generator, verbose=1)

    # 4. Inverse Transform untuk Mendapatkan Kecepatan Aktual (mm/min)
    y_pred_log = preprocessor.target_scaler.inverse_transform(y_pred_scaled)
    raw_predicted_feedrate = np.maximum(1.0, np.expm1(y_pred_log).flatten())

    # --- SAFEGUARD: PHYSICS-INFORMED CLIPPING & KINEMATIC CORRECTION ---
    cmd_f_limits = df_parsed['Cmd_F'].values.copy()

    is_rotary = (df_parsed['Delta_Rot'].values > 1e-4) | (df_parsed['Tool_Vector_Delta'].values > 1e-4)
    is_micro = df_parsed['Delta_3D'].values < 0.5

    # Batas Bawah: Hybrid Clamping (1% untuk gerak 5-Axis/rotasi atau Micro-Block, 10% untuk gerak linear normal)
    min_multiplier = np.where(is_rotary | is_micro, 0.01, 0.10)
    min_feedrate_limits = np.maximum(min_multiplier * cmd_f_limits, 1.0)

    predicted_feedrate = np.clip(raw_predicted_feedrate, min_feedrate_limits, cmd_f_limits)

    # KOREKSI FISIKA 1: Garis Lurus Panjang G01 3-Axis (Delta_3D > 5.0 mm, tanpa rotasi meja, sudut belok < 15 derajat)
    # Pada segmen lurus panjang tanpa rotasi sumbu B/C, mesin CNC mencapai kecepatan steady-state (~98.5% Cmd_F)
    is_long_straight_g01 = (
        (df_parsed['Is_G01'].values == 1) &
        (df_parsed['Delta_3D'].values > 5.0) &
        (~is_rotary) &
        (df_parsed['Sharpness_Angle'].values < 0.2618)  # < 15 derajat (0.2618 rad)
    )
    predicted_feedrate = np.where(
        is_long_straight_g01,
        np.maximum(predicted_feedrate, 0.985 * cmd_f_limits),
        predicted_feedrate
    )

    # KOREKSI FISIKA 2: Kecepatan Vektor Efektif G00 Rapid Jarak Menengah-Jauh (Delta_3D > 10.0 mm)
    is_rapid_block = (
        (df_parsed['Is_G01'].values == 0) &
        (df_parsed['Is_G02'].values == 0) &
        (df_parsed['Is_G03'].values == 0)
    )
    is_traori_init_sync = (
        is_rapid_block &
        (df_parsed['Is_Traori'].values == 1) &
        (df_parsed['Delta_3D'].values > 50.0) &
        (df_parsed['Tool_Vector_Delta'].values > 0.05) &
        (np.arange(len(df_parsed)) < 15)
    )
    is_long_rapid = is_rapid_block & (~is_traori_init_sync) & (df_parsed['Delta_3D'].values > 10.0)
    predicted_feedrate = np.where(
        is_long_rapid,
        np.maximum(predicted_feedrate, 23000.0),
        predicted_feedrate
    )
    # -------------------------------------------------------------------

    # 5. Integrasi Kinematika Fisik & Perhitungan Durasi per Kategori Gerakan
    df_parsed['Predicted_Feedrate_mm_min'] = predicted_feedrate

    has_linear_motion = df_parsed['Delta_3D'].values > 1e-4
    has_rotary_only = (~has_linear_motion) & (df_parsed['Delta_Rot'].values > 1e-4)
    dwell_times = df_parsed.get('G04_Dwell_Time', pd.Series([0.0] * len(df_parsed))).values

    block_durations_sec = np.where(
        is_traori_init_sync,
        0.40,  # Fallback jika masih ada blok sinkronisasi awal TRAORI (~0.40 detik)
        np.where(
            has_linear_motion,
            (df_parsed['Delta_3D'].values / predicted_feedrate) * 60.0,
            np.where(
                has_rotary_only,
                (df_parsed['Delta_Rot'].values / 5000.0) * 60.0,  # Kecepatan indexing rotasi murni B/C (deg/min)
                dwell_times
            )
        )
    )
    df_parsed['Estimated_Duration_Sec'] = block_durations_sec

    total_time_sec = float(df_parsed['Estimated_Duration_Sec'].sum())
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
