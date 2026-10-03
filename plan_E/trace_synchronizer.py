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
        Membersihkan trace dan mengatribusikan block number negatif (CYCLE800 swiveling & MCALL).
        """
        df = df_trace.copy()

        df.columns = df.columns.str.strip()

        def find_column_by_substrings(substrings: List[str]) -> str:
            for col in df.columns:
                if any(sub in col for sub in substrings):
                    return col
            return None

        col_line = find_column_by_substrings(['actLineNumber', 'f1\\s1', 'f1/s1'])
        if not col_line:
            col_line = find_column_by_substrings(['f1', 's1'])
            if not col_line:
                raise KeyError(f"Kolom Line Number tidak ditemukan di file trace! Kolom yang tersedia: {list(df.columns)}")
        df.rename(columns={col_line: 'actLineNumber'}, inplace=True)

        col_x = find_column_by_substrings(['f2\\s2', 'f2/s2', 'f2', 'actToolBasePos[0]'])
        col_y = find_column_by_substrings(['f3\\s3', 'f3/s3', 'f3', 'actToolBasePos[1]'])
        col_z = find_column_by_substrings(['f4\\s4', 'f4/s4', 'f4', 'actToolBasePos[2]'])
        col_b = find_column_by_substrings(['f5\\s5', 'f5/s5', 'f5', 'actToolBasePos[3]'])
        col_c = find_column_by_substrings(['f6\\s6', 'f6/s6', 'f6', 'actToolBasePos[4]'])

        delta_x = pd.to_numeric(df[col_x], errors='coerce').diff().abs().fillna(0) if col_x else pd.Series(0, index=df.index)
        delta_y = pd.to_numeric(df[col_y], errors='coerce').diff().abs().fillna(0) if col_y else pd.Series(0, index=df.index)
        delta_z = pd.to_numeric(df[col_z], errors='coerce').diff().abs().fillna(0) if col_z else pd.Series(0, index=df.index)
        delta_b = pd.to_numeric(df[col_b], errors='coerce').diff().abs().fillna(0) if col_b else pd.Series(0, index=df.index)
        delta_c = pd.to_numeric(df[col_c], errors='coerce').diff().abs().fillna(0) if col_c else pd.Series(0, index=df.index)

        is_moving = (delta_x > 1e-4) | (delta_y > 1e-4) | (delta_z > 1e-4) | (delta_b > 1e-4) | (delta_c > 1e-4)

        mapped_blocks = []
        last_valid_n_number = None

        for idx, row in df.iterrows():
            try:
                raw_line = int(float(row['actLineNumber']))
            except (ValueError, TypeError):
                mapped_blocks.append("IDLE")
                continue

            moving = is_moving.iloc[idx]

            if raw_line > 0:
                mapped_blocks.append(str(raw_line))
                last_valid_n_number = str(raw_line)

            elif raw_line < 0 and moving:
                # Transisi CYCLE800 / MCALL
                mapped_blocks.append(f"SUB_{last_valid_n_number}" if last_valid_n_number else "INIT_IDLE")
            else:
                mapped_blocks.append("IDLE")

        df['mapped_block'] = mapped_blocks
        df_valid = df[~df['mapped_block'].isin(["IDLE", "INIT_IDLE"])].copy()

        # Override the actLineNumber so that the rest of the algorithms treat SUB blocks as part of the parent!
        # This is critical so that `line_end_indices` covers the entire drilling sequence.
        def remap_line(x):
            if str(x).startswith("SUB_"):
                return int(str(x).replace("SUB_", ""))
            elif str(x).isdigit():
                return int(x)
            return x

        df_valid['actLineNumber'] = df_valid['mapped_block'].apply(remap_line)
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
                    v_avg_raw = float(np.mean(v_slice)) if len(v_slice) > 0 else 0.0
                else:
                    dur = 0.0
                    v_avg_raw = 0.0

                cmd_f_limit = df_gcode.loc[idx, 'Cmd_F']
                if pd.isnull(cmd_f_limit): cmd_f_limit = 20000.0

                # 1. Clamp Feedrate
                v_avg = min(v_avg_raw, cmd_f_limit, 20000.0)

                rot_3d = float(df_gcode.loc[idx, 'Delta_Rot']) if 'Delta_Rot' in df_gcode.columns else 0.0
                if pd.isnull(rot_3d): rot_3d = 0.0

                # 2. Non-motion clamp
                if d_3d <= 1e-4 and rot_3d <= 1e-4:
                    v_avg = cmd_f_limit

                # 3. Realistic duration guard
                if d_3d > 1e-4:
                    eff_feed = v_avg if v_avg > 0 else cmd_f_limit
                    if eff_feed > 0:
                        dur = max(dur, (d_3d / eff_feed) * 60.0)

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
            if block_id > 0:
                last_valid_block_id = block_id
            else:
                block_id = last_valid_block_id

            target_xyz = np.array([row.get('Tgt_X', 0.0), row.get('Tgt_Y', 0.0), row.get('Tgt_Z', 0.0)])
            d_3d = float(row.get('Delta_3D', 0.0))

            ref_block = get_next_valid_line(block_id)
            end_bound_idx = line_end_indices.get(ref_block, len(trace_coords)-1)
            if end_bound_idx < last_actual_idx:
                 end_bound_idx = last_actual_idx

            end_search_idx = min(end_bound_idx + 10, len(trace_coords))
            start_search_idx = min(last_actual_idx, end_search_idx)

            search_window = trace_coords[start_search_idx:end_search_idx]

            if len(search_window) == 0:
                actual_idx = start_search_idx
            else:
                distances = np.linalg.norm(search_window - target_xyz, axis=1)
                min_local_idx = np.argmin(distances)
                actual_idx = start_search_idx + min_local_idx

            if actual_idx > last_actual_idx and actual_idx <= len(trace_times):
                dur = float(trace_times[actual_idx] - trace_times[last_actual_idx])
                v_slice = trace_velocs[last_actual_idx:actual_idx]
                mean_feed_raw = float(np.mean(v_slice)) if len(v_slice) > 0 else 0.0
            else:
                dur = 0.0
                safe_idx = min(actual_idx, len(trace_velocs) - 1)
                mean_feed_raw = float(trace_velocs[safe_idx])

            cmd_f_limit = row.get('Cmd_F', 20000.0)
            if pd.isnull(cmd_f_limit):
                cmd_f_limit = 20000.0

            mean_feed = min(mean_feed_raw, cmd_f_limit, 20000.0)

            rot_3d = float(row.get('Delta_Rot', 0.0))
            if pd.isnull(rot_3d): rot_3d = 0.0

            if d_3d <= 1e-4 and rot_3d <= 1e-4:
                mean_feed = cmd_f_limit

            if d_3d > 1e-4:
                eff_feed = mean_feed if mean_feed > 0 else cmd_f_limit
                if eff_feed > 0:
                    dur = max(dur, (d_3d / eff_feed) * 60.0)

            est_dur = 0.0 if mean_feed <= 0 else (d_3d / (mean_feed / 60.0))

            iloc_idx = df_gcode.index.get_loc(idx)
            durations[iloc_idx] = dur
            feedrates[iloc_idx] = mean_feed
            estimasi_durasis[iloc_idx] = est_dur

            last_actual_idx = actual_idx

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
