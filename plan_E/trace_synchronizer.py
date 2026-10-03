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
        Algoritma Hibrida Lanjutan (Spasial + Batasan Sinyal Blok + Distribusi Proporsional).
        - Zona 1 (Persiapan) & Zona 3 (Postposition): Distribusi jarak proporsional.
        - Zona 2 (Cutting): Hibrida Spasial dengan Batasan Sinyal Blok.
        """
        df_gcode = df_parsed_gcode.copy()

        # Fallback jika Segment tidak ada (kompatibilitas cache lama)
        if 'Segment' not in df_gcode.columns:
            df_gcode['Segment'] = 2

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

        # Prepare target arrays aligned with df_gcode
        durations = np.zeros(len(df_gcode), dtype=float)
        feedrates = np.zeros(len(df_gcode), dtype=float)
        estimasi_durasis = np.zeros(len(df_gcode), dtype=float)

        # Helper: Proporsional untuk Segment 1 dan 3
        def process_proportional_zone(group_indices, start_idx_bound, end_idx_bound):
            if len(group_indices) == 0:
                return start_idx_bound

            tot_samples = max(0, end_idx_bound - start_idx_bound)
            tot_dist = df_gcode.loc[group_indices, 'Delta_3D'].sum()
            curr_slice_start = start_idx_bound

            for i_local, idx in enumerate(group_indices):
                d_3d = float(df_gcode.loc[idx, 'Delta_3D'])

                if tot_dist > 0:
                    porsi = d_3d / tot_dist
                else:
                    porsi = 1.0 / len(group_indices)

                n_samples = int(round(porsi * tot_samples))
                curr_slice_end = min(curr_slice_start + n_samples, end_idx_bound)

                # Paksa blok terakhir mengambil semua sisa sampel
                if i_local == len(group_indices) - 1:
                    curr_slice_end = end_idx_bound

                safe_start = min(curr_slice_start, len(trace_times) - 1)
                safe_end = min(curr_slice_end, len(trace_times) - 1)
                if safe_end > safe_start:
                    dur = float(trace_times[safe_end] - trace_times[safe_start])
                    v_slice = trace_velocs[safe_start:safe_end]
                    v_avg = float(np.mean(v_slice)) if len(v_slice) > 0 else 0.0
                else:
                    dur = 0.0
                    v_avg = 0.0

                est_dur = 0.0 if v_avg <= 0 else (d_3d / (v_avg / 60.0))

                iloc_idx = df_gcode.index.get_loc(idx)
                durations[iloc_idx] = dur
                feedrates[iloc_idx] = v_avg
                estimasi_durasis[iloc_idx] = est_dur

                curr_slice_start = curr_slice_end

            return curr_slice_start

        # --- Pembagian Zona ---
        seg1_indices = df_gcode[df_gcode['Segment'] == 1].index
        seg2_indices = df_gcode[df_gcode['Segment'] == 2].index
        seg3_indices = df_gcode[df_gcode['Segment'] == 3].index

        last_actual_idx = 0

        # --- ZONA 1 (Persiapan) ---
        if len(seg1_indices) > 0:
            if len(seg2_indices) > 0:
                first_seg2_block = int(df_gcode.loc[seg2_indices[0], 'N_Number'])
                # Forward-fill if anonymous block
                if first_seg2_block <= 0:
                     first_seg2_block = max(1, int(df_gcode.loc[seg2_indices[0]-1, 'N_Number']))

                ref_first_seg2 = get_next_valid_line(first_seg2_block)
                if ref_first_seg2 in valid_trace_lines:
                    seg1_end_bound = np.where(trace_linenum == ref_first_seg2)[0][0]
                else:
                    seg1_end_bound = min(100, len(trace_coords)-1)
            else:
                seg1_end_bound = len(trace_coords)-1

            last_actual_idx = process_proportional_zone(seg1_indices, 0, seg1_end_bound)

        # --- ZONA 2 (Cutting - Hibrida Spasial) ---
        last_valid_block_id = 1
        for idx in seg2_indices:
            row = df_gcode.loc[idx]
            block_id = int(row['N_Number'])
            is_mcall = int(row.get('Is_MCALL_Sub', 0))

            # 1. Forward-fill N_Number untuk menangani blok MCALL/turunan yang tidak bernomor
            if block_id > 0:
                last_valid_block_id = block_id
            else:
                block_id = last_valid_block_id

            if is_mcall == 1:
                # KASUS A: BLOK PECAHAN MCALL (Gunakan Interpolasi Teoritis)
                # Ambil Target Feedrate dari Command (karena data Trace terkompresi)
                cmd_feed = float(row.get('Cmd_F', 0.0))
                dist_3d = float(row.get('Delta_3D', 0.0))
                is_dwell = int(row.get('Is_G01', -1)) == -1 and int(row.get('Is_G02', -1)) == -1 and int(row.get('Is_G03', -1)) == -1 # Simulasi cek dwell

                # Cek khusus untuk G04 Dwell Time di dalam siklus MCALL
                if float(row.get('G04_Dwell_Time', 0.0)) > 0.0:
                    durasi_trace = float(row.get('G04_Dwell_Time'))
                    mean_feedrate = 0.0
                else:
                    mean_feedrate = cmd_feed
                    # Hitung durasi secara proporsional berdasar jarak dan Cmd_F
                    durasi_trace = (dist_3d / (cmd_feed / 60.0)) if cmd_feed > 0.0 else 0.0

                # Tidak perlu update last_actual_idx agar pencarian Euclidean selanjutnya tetap sinkron
                actual_idx = last_actual_idx

                dur = durasi_trace
                mean_feed = mean_feedrate
                d_3d = dist_3d

            else:
                # KASUS B: BLOK STANDAR (Gunakan Hibrida Spasial - Batas Sinyal)
                target_xyz = np.array([row.get('Tgt_X', 0.0), row.get('Tgt_Y', 0.0), row.get('Tgt_Z', 0.0)])
                d_3d = float(row.get('Delta_3D', 0.0))

                ref_block = get_next_valid_line(block_id)
                end_bound_idx = line_end_indices.get(ref_block, len(trace_coords)-1)
                end_search_idx = min(end_bound_idx + 10, len(trace_coords))

                start_search_idx = min(last_actual_idx, end_search_idx)

                if end_search_idx <= start_search_idx:
                     end_search_idx = start_search_idx + 1

                search_window = trace_coords[start_search_idx:end_search_idx]

                if len(search_window) == 0:
                    actual_idx = start_search_idx
                else:
                    distances = np.linalg.norm(search_window - target_xyz, axis=1)
                    min_local_idx = np.argmin(distances)
                    actual_idx = start_search_idx + min_local_idx

                durasi_trace = trace_times[actual_idx] - trace_times[last_actual_idx]

                if actual_idx > last_actual_idx:
                    v_slice = trace_velocs[last_actual_idx:actual_idx]
                    mean_feedrate = np.mean(v_slice) if len(v_slice) > 0 else 0.0
                else:
                    safe_idx = min(actual_idx, len(trace_velocs)-1)
                    mean_feedrate = trace_velocs[safe_idx]

                last_actual_idx = actual_idx
                dur = durasi_trace
                mean_feed = mean_feedrate

            est_dur = 0.0 if mean_feed <= 0 else (d_3d / (mean_feed / 60.0))

            iloc_idx = df_gcode.index.get_loc(idx)
            durations[iloc_idx] = dur
            feedrates[iloc_idx] = mean_feed
            estimasi_durasis[iloc_idx] = est_dur

        # --- ZONA 3 (Postposition) ---
        if len(seg3_indices) > 0:
            process_proportional_zone(seg3_indices, last_actual_idx, len(trace_coords) - 1)

        df_gcode['Duration_Sec'] = list(durations)
        df_gcode['Target_Feedrate'] = list(feedrates)
        df_gcode['Estimasi_Durasi_Teoritis_s'] = list(estimasi_durasis)

        return df_gcode

if __name__ == "__main__":
    # Contoh verifikasi modul
    syncer = SinuTrainSynchronizer()
    print("[INFO] Trace Synchronizer Module siap digunakan.")
