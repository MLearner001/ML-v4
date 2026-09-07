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
        Menghubungkan trace per blok dan menghitung Target Feedrate Harmonik (Y)
        serta menangani micro-blocks sub-4ms via Distance-Weighted Spatial Interpolation.
        """
        df_gcode = df_parsed_gcode.copy()

        # 1. Hitung jumlah tick dan rata-rata kecepatan terukur dari trace
        trace_counts = df_trace_valid['mapped_block'].value_counts().to_dict()

        # 2. Identifikasi blok yang tereksekusi langsung vs micro-blocks yang terlewati
        durations = []
        target_feedrates = []

        i = 0
        n_blocks = len(df_gcode)

        while i < n_blocks:
            cluster_indices = [i]

            # Fungsi bantuan untuk mendapatkan kunci pencocokan
            def get_sync_key(idx):
                row_data = df_gcode.iloc[idx]
                n = str(int(row_data['N_Number'])) if row_data['N_Number'] != -1 else "-1"
                if row_data.get('Is_Cycle800', 0) == 1:
                    return f"C800_{n}"
                return n

            sync_key = get_sync_key(i)
            ticks = trace_counts.get(sync_key, 0)

            j = i + 1

            if ticks > 0:
                # KASUS A: Blok tereksekusi normal / Toolpath Panjang
                # Cari sub-blocks yang memiliki N_Number yang sama (misal hasil ekspansi MCALL)
                # atau blok tanpa N_Number (-1) yang menempel setelahnya.
                while j < n_blocks:
                    next_sync_key = get_sync_key(j)
                    row_j = df_gcode.iloc[j]
                    if next_sync_key == sync_key or row_j['N_Number'] == -1:
                        cluster_indices.append(j)
                        j += 1
                    else:
                        break

                cluster_dt = ticks * self.dt

            else:
                # KASUS B: Micro-blocks (0 ticks) - SinuTrain melompati blok ini
                # Lakukan Look-Ahead: Gabungkan blok ini dengan blok-blok berikutnya
                # hingga menemukan blok "Anchor" yang terekam di trace (>0 ticks).
                anchor_ticks = 0
                while j < n_blocks:
                    cluster_indices.append(j)
                    next_sync_key = get_sync_key(j)
                    anchor_ticks = trace_counts.get(next_sync_key, 0)

                    if anchor_ticks > 0:
                        # Anchor ditemukan!
                        # Ambil juga sub-blocks dari anchor ini agar menjadi satu kluster utuh
                        anchor_key = next_sync_key
                        k = j + 1
                        while k < n_blocks:
                            k_key = get_sync_key(k)
                            row_k = df_gcode.iloc[k]
                            if k_key == anchor_key or row_k['N_Number'] == -1:
                                cluster_indices.append(k)
                                k += 1
                            else:
                                break
                        j = k  # Update j ke akhir sub-blocks anchor
                        break
                    else:
                        j += 1

                # Total durasi untuk seluruh micro-blocks + anchor adalah ticks dari anchor
                # Jika sudah di akhir file dan tidak ada anchor, beri minimal 1 interval 4ms
                cluster_dt = anchor_ticks * self.dt if anchor_ticks > 0 else 1.0 * self.dt

            # --- DISTRIBUSI WAKTU PROPORSIONAL (Distance-Weighted Interpolation) ---
            cluster_dists = [
                df_gcode.iloc[k]['Delta_3D'] if df_gcode.iloc[k]['Delta_3D'] > 1e-4 else df_gcode.iloc[k]['Delta_Rot']
                for k in cluster_indices
            ]
            total_cluster_dist = sum(cluster_dists)

            if total_cluster_dist > 1e-6:
                # Feedrate harmonik rata-rata dari kluster
                f_group = (total_cluster_dist / cluster_dt) * 60.0 if cluster_dt > 0 else df_gcode.iloc[cluster_indices[0]]['Cmd_F']

                # Bagikan waktu secara proporsional berdasarkan jarak masing-masing
                for k, d in zip(cluster_indices, cluster_dists):
                    weight = d / total_cluster_dist
                    t_sub = weight * cluster_dt
                    durations.append(t_sub)
                    target_feedrates.append(f_group)
            else:
                # Gerakan diam murni (misal logika G54, tool change, dwell)
                for k in cluster_indices:
                    t_sub = cluster_dt / len(cluster_indices)
                    durations.append(t_sub)
                    # Fallback ke commanded feedrate jika tidak ada jarak
                    target_feedrates.append(df_gcode.iloc[k]['Cmd_F'])

            i = j  # Lompat ke blok setelah kluster diproses

        df_gcode['Duration_Sec'] = durations
        df_gcode['Target_Feedrate'] = target_feedrates

        return df_gcode


if __name__ == "__main__":
    # Contoh verifikasi modul
    syncer = SinuTrainSynchronizer()
    print("[INFO] Trace Synchronizer Module siap digunakan.")
