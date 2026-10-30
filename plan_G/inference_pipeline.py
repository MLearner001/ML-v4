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

    # --- KOREKSI FISIKA 1: Garis Lurus Panjang & Rantai Lurus/Mulus Terdiskritisasi CAM (3-Axis G01) ---
    rolling_d3d_11 = df_parsed['Delta_3D'].rolling(window=11, center=True, min_periods=1).sum().values
    rolling_max_angle_7 = df_parsed['Sharpness_Angle'].rolling(window=7, center=True, min_periods=1).max().values
    rolling_mean_angle_7 = df_parsed['Sharpness_Angle'].rolling(window=7, center=True, min_periods=1).mean().values

    # Batas kecepatan fisika Newtonian sepanjang jendela 11 blok (mm/min) untuk memastikan jarak akselerasi cukup
    window_kinematic_limit = np.sqrt(1000.0 * np.maximum(0.0, rolling_d3d_11)) * 60.0
    has_enough_accel_dist = (rolling_d3d_11 > 5.0) & (window_kinematic_limit > 2.2 * cmd_f_limits)

    # 1a. Segmen lurus tunggal (> 5.0 mm, sudut belok < 15 derajat)
    is_single_long_straight = (
        (df_parsed['Delta_3D'].values > 5.0) &
        (df_parsed['Sharpness_Angle'].values < 0.2618)
    )
    # 1b. Rantai garis lurus yang dicacah CAM menjadi segmen 0.25 - 5.0 mm
    is_discretized_straight_chain = (
        (df_parsed['Delta_3D'].values > 0.25) &
        has_enough_accel_dist &
        (df_parsed['Sharpness_Angle'].values < 0.0524) &  # < 3 derajat pada blok saat ini
        (rolling_max_angle_7 < 0.0873)                    # < 5 derajat di 7 blok sekitar
    )
    # 1c. Rantai kontur 3-axis mulus pada toleransi CYCLE832 >= 0.01
    is_smooth_contour_chain = (
        (df_parsed['Delta_3D'].values > 0.20) &
        (rolling_d3d_11 > 3.0) &
        (window_kinematic_limit > 1.8 * cmd_f_limits) &
        (df_parsed['C832_Tol'].values >= 0.01) &
        (rolling_mean_angle_7 < 0.0698) &                 # Rata-rata sudut < 4 derajat
        (rolling_max_angle_7 < 0.1396)                    # Maksimum sudut < 8 derajat
    )

    is_g01_non_rotary = (df_parsed['Is_G01'].values == 1) & (~is_rotary)
    is_arc_non_rotary = ((df_parsed['Is_G02'].values == 1) | (df_parsed['Is_G03'].values == 1)) & (~is_rotary)

    # 1d. Busur melingkar panjang atau Helical Drilling (G02/G03 dengan TURN > 0 atau panjang busur > 10 mm)
    is_helical_or_long_arc = (
        is_arc_non_rotary &
        ((df_parsed['Turn_Count'].values > 0) | (df_parsed['Delta_3D'].values > 10.0))
    )

    predicted_feedrate = np.where(
        (is_g01_non_rotary & (is_single_long_straight | is_discretized_straight_chain)) | is_helical_or_long_arc,
        np.maximum(predicted_feedrate, 0.985 * cmd_f_limits),
        np.where(
            is_g01_non_rotary & is_smooth_contour_chain,
            np.maximum(predicted_feedrate, 0.920 * cmd_f_limits),
            predicted_feedrate
        )
    )

    # --- KOREKSI FISIKA 2: Kecepatan Vektor Efektif G00 Rapid Jarak Menengah-Jauh (Delta_3D > 10.0 mm) ---
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
    is_long_rapid = is_rapid_block & (~is_traori_init_sync) & (~is_rotary) & (df_parsed['Delta_3D'].values > 10.0)
    predicted_feedrate = np.where(
        is_long_rapid,
        np.maximum(predicted_feedrate, 23000.0),
        predicted_feedrate
    )
    # -------------------------------------------------------------------

    # 5. Integrasi Kinematika Fisik & Perhitungan Durasi per Kategori Gerakan
    df_parsed['Predicted_Feedrate_mm_min'] = predicted_feedrate

    has_linear_motion = df_parsed['Delta_3D'].values > 1e-4
    dwell_times = df_parsed.get('G04_Dwell_Time', pd.Series([0.0] * len(df_parsed))).values

    linear_duration_sec = np.where(
        has_linear_motion,
        (df_parsed['Delta_3D'].values / predicted_feedrate) * 60.0,
        0.0
    )

    # KOREKSI FISIKA 3: Batas Durasi Minimum Sumbu Putar B/C (Rotary Kinematic Time Floor)
    max_delta_bc = np.maximum(
        df_parsed['Delta_B'].values,
        np.maximum(df_parsed['Delta_C'].values, df_parsed['Delta_Rot'].values)
    )
    eff_rotary_speed_deg_min = np.where(is_rapid_block, 5000.0, 1400.0)
    rotary_floor_sec = np.where(
        is_rotary,
        (max_delta_bc / eff_rotary_speed_deg_min) * 60.0,
        0.0
    )

    block_durations_sec = np.where(
        is_traori_init_sync,
        0.40,  # Fallback jika masih ada blok sinkronisasi awal TRAORI (~0.40 detik)
        np.where(
            has_linear_motion | is_rotary,
            np.maximum(linear_duration_sec, rotary_floor_sec),
            dwell_times
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
