"""
trace_synchronizer.py

Tahap 2: Sinkronisasi data trace 4ms SinuTrain dengan G-code hasil parsing Tahap 1.
Menerapkan Harmonic Target, Distance-Weighted Interpolation, dan penanganan transisi CYCLE800.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class SinuTrainSynchronizer:
    def __init__(self, sample_interval_sec: float = 0.004):
        self.dt = sample_interval_sec  # 4ms = 0.004 s

    def clean_and_attribute_trace(self, df_trace: pd.DataFrame, gcode_blocks: List[str]) -> pd.DataFrame:
        """
        Membersihkan trace dan mengatribusikan block number negatif (CYCLE800 swiveling).
        """
        df = df_trace.copy()

        # Standarisasi nama kolom trace SinuTrain jika diperlukan
        # Bersihkan spasi kosong di kolom jika belum
        df.columns = df.columns.str.strip()

        # Kolom utama: actLineNumber, f2/s2 (X), f3/s3 (Y), f4/s4 (Z), f5/s5 (B), f6/s6 (C)
        # Pada beberapa export, nama kolom mengandung path panjang seperti '/Channel/!SPARP/actLineNumber [u1  1]'

        def find_column_by_substrings(substrings: List[str]) -> str:
            for col in df.columns:
                if any(sub in col for sub in substrings):
                    return col
            return None

        # Temukan kolom actLineNumber
        col_line = find_column_by_substrings(['actLineNumber', 'f1\\s1', 'f1/s1'])
        if not col_line:
            # Fallback agresif
            col_line = find_column_by_substrings(['f1', 's1'])
            if not col_line:
                raise KeyError(f"Kolom Line Number tidak ditemukan di file trace! Kolom yang tersedia: {list(df.columns)}")
        df.rename(columns={col_line: 'actLineNumber'}, inplace=True)

        # Temukan kolom f5/s5 (B) dan f6/s6 (C)
        # Pada file Siemens .csv raw sering digunakan backslash
        col_b = find_column_by_substrings(['f5\\s5', 'f5/s5', 'f5', 'actToolBasePos[3]'])
        col_c = find_column_by_substrings(['f6\\s6', 'f6/s6', 'f6', 'actToolBasePos[4]'])

        # Hitung diff posisi B dan C (Numerical Position Differentiation)
        if col_b and col_b in df.columns:
            delta_b = pd.to_numeric(df[col_b], errors='coerce').diff().abs().fillna(0)
        else:
            delta_b = pd.Series(0, index=df.index)

        if col_c and col_c in df.columns:
            delta_c = pd.to_numeric(df[col_c], errors='coerce').diff().abs().fillna(0)
        else:
            delta_c = pd.Series(0, index=df.index)

        is_rotary_moving = (delta_b > 1e-4) | (delta_c > 1e-4)

        mapped_blocks = []

        # SinuTrain actLineNumber corresponds perfectly to G-Code N_Number
        last_valid_n_number = None

        for idx, row in df.iterrows():
            try:
                # Handle possible NaN / empty string lines
                raw_line = int(float(row['actLineNumber']))
            except (ValueError, TypeError):
                mapped_blocks.append("IDLE")
                continue

            rot_moving = is_rotary_moving.iloc[idx]

            if raw_line > 0:
                # G-Code line explicitly mapped
                mapped_blocks.append(str(raw_line))
                last_valid_n_number = str(raw_line)

            elif raw_line < 0 and rot_moving:
                # Transisi CYCLE800 / Orientasi Bidang: Atribusikan ke blok parent CYCLE800 (the positive N_number)
                mapped_blocks.append(f"C800_{last_valid_n_number}" if last_valid_n_number else "INIT_IDLE")
            else:
                # Idle tanpa pergerakan signifikan
                mapped_blocks.append("IDLE")

        df['mapped_block'] = mapped_blocks

        # Buang baris trace yang tergolong IDLE murni (tidak ada eksekusi program benda kerja)
        df_valid = df[~df['mapped_block'].isin(["IDLE", "INIT_IDLE"])].copy()
        return df_valid

    def match_and_calculate_targets(self, df_parsed_gcode: pd.DataFrame, df_trace_valid: pd.DataFrame) -> pd.DataFrame:
        """
        Algoritma Hibrida (Spasial + Batasan Sinyal Blok).
        Mencari posisi XYZ terdekat di Trace SinuTrain berdasarkan interval waktu
        dari Sinyal Blok (actLineNumber).
        """
        df_gcode = df_parsed_gcode.copy()

        dt = self.dt
        trace_times = df_trace_valid['time'].to_numpy() if 'time' in df_trace_valid.columns else df_trace_valid.index.to_numpy() * dt

        x_col = next((c for c in df_trace_valid.columns if 'f2' in c or 'X' in c), None)
        y_col = next((c for c in df_trace_valid.columns if 'f3' in c or 'Y' in c), None)
        z_col = next((c for c in df_trace_valid.columns if 'f4' in c or 'Z' in c), None)

        if not (x_col and y_col and z_col):
            raise KeyError(f"Kolom X, Y, Z tidak ditemukan di trace. Tersedia: {list(df_trace_valid.columns)}")

        trace_coords = df_trace_valid[[x_col, y_col, z_col]].to_numpy()

        vel_col = next((c for c in df_trace_valid.columns if 'f7' in c or 'V' in c), None)
        trace_velocs = df_trace_valid[vel_col].to_numpy() if vel_col else np.zeros(len(trace_coords))

        trace_linenum = df_trace_valid['actLineNumber'].to_numpy()

        valid_trace_lines = np.unique(trace_linenum)
        valid_trace_lines = valid_trace_lines[valid_trace_lines > 0]
        valid_trace_lines.sort()

        def get_next_valid_line(target_n):
            idx = np.searchsorted(valid_trace_lines, target_n)
            if idx < len(valid_trace_lines):
                return valid_trace_lines[idx]
            if len(valid_trace_lines) > 0:
                return valid_trace_lines[-1]
            return target_n

        line_end_indices = {}
        for ln in valid_trace_lines:
            line_end_indices[ln] = np.where(trace_linenum == ln)[0][-1]

        durations = []
        feedrates = []
        last_actual_idx = 0
        last_valid_block_id = 1

        for idx, row in df_gcode.iterrows():
            block_id = int(row['N_Number'])

            if block_id > 0:
                last_valid_block_id = block_id
            else:
                block_id = last_valid_block_id

            target_xyz = np.array([row.get('Tgt_X', 0.0), row.get('Tgt_Y', 0.0), row.get('Tgt_Z', 0.0)])
            d_3d = float(row.get('Delta_3D', 0.0))
            d_rot = float(row.get('Delta_Rot', 0.0))
            cmd_f_limit = float(row.get('Cmd_F', 20000.0))
            if cmd_f_limit <= 0.0:
                cmd_f_limit = 20000.0

            # --- Sinyal Blok sebagai Pagar Batas ---
            ref_block = get_next_valid_line(block_id)
            end_bound_idx = line_end_indices.get(ref_block, len(trace_coords)-1)
            # Mencegah end_bound_idx lebih kecil dari last_actual_idx
            if end_bound_idx < last_actual_idx:
                 end_bound_idx = last_actual_idx

            end_search_idx = min(end_bound_idx + 10, len(trace_coords))

            # Kunci Progresif: pencarian dimulai dari last_actual_idx
            start_search_idx = last_actual_idx

            if end_search_idx <= start_search_idx:
                 end_search_idx = start_search_idx + 1

            search_window = trace_coords[start_search_idx:end_search_idx]

            # Pencarian Spasial di dalam Pagar
            if len(search_window) == 0:
                actual_idx = start_search_idx
            else:
                distances = np.linalg.norm(search_window - target_xyz, axis=1)
                min_local_idx = np.argmin(distances)
                actual_idx = start_search_idx + min_local_idx

            # Hitung tick (selisih indeks)
            ticks = actual_idx - last_actual_idx

            # --- PROTEKSI KASUS B (Penggabungan Blok Rapid) ---
            if d_3d > 1.0 and ticks <= 2:
                # Terlalu sedikit tick untuk jarak besar, ini anomali / tidak ada eksekusi nyata
                f_raw = cmd_f_limit
            else:
                if ticks > 0:
                    v_slice = trace_velocs[last_actual_idx:actual_idx]
                    f_raw = float(np.mean(v_slice))
                else:
                    safe_idx = min(actual_idx, len(trace_velocs)-1)
                    f_raw = float(trace_velocs[safe_idx])

            t_sub = float(max(ticks * dt, dt))

            # --- PHYSICAL GUARD CLAMPING & PROTEKSI DURASI ---
            if d_3d < 1e-4 and d_rot < 1e-4:
                # Blok non-motion, tetapkan durasi dan feedrate absolut tanpa mendistorsi spasial berikutnya
                f_clamped = cmd_f_limit
                t_sub = float(max(ticks * dt, dt))
            else:
                f_clamped = min(f_raw, cmd_f_limit, 20000.0)
                if f_clamped < 1e-4:
                    f_clamped = min(cmd_f_limit, 20000.0)
                # Proteksi Durasi minimal berdasarkan jarak
                kinematic_t = (d_3d / f_clamped) * 60.0
                t_sub = float(max(t_sub, kinematic_t))

            # Pengaman Ekstra absolut agar durasi tidak pernah di bawah dt
            if t_sub < dt:
                t_sub = dt

            durations.append(t_sub)
            feedrates.append(f_clamped)
            last_actual_idx = actual_idx

        df_gcode['Duration_Sec'] = durations
        df_gcode['Target_Feedrate'] = feedrates

        feedrates_arr = np.array(feedrates)
        estimasi_durasi_teoritis = np.where(
            feedrates_arr > 0.0,
            (df_gcode['Delta_3D'].values / np.maximum(feedrates_arr, 1e-6)) * 60.0,
            0.0
        )
        df_gcode['Estimasi_Durasi_Teoritis_s'] = estimasi_durasi_teoritis

        return df_gcode


if __name__ == "__main__":
    # Contoh verifikasi modul
    syncer = SinuTrainSynchronizer()
    print("[INFO] Trace Synchronizer Module siap digunakan.")
