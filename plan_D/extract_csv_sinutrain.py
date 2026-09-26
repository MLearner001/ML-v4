import os
import glob
import pandas as pd
import numpy as np
import re

def get_mpf_blocks_with_distance(mpf_filepath):
    """Mengekstrak urutan Block Number (N) dan Jarak Pergerakan dari file G-Code (.mpf)."""
    blocks = []

    # Memori koordinat mesin virtual
    x, y, z, b, c = 0.0, 0.0, 0.0, 0.0, 0.0
    a3, b3, c3 = 0.0, 0.0, 1.0

    # Status memori mode gerak (Modal)
    current_motion_mode = "G00"

    with open(mpf_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # Hapus komentar (;) agar Regex fokus ke instruksi fisik
            if ";" in line:
                line = line.split(";")[0].strip()
            if not line:
                continue

            # Deteksi perubahan Modal Motion
            if re.search(r"\bG0*0\b", line, re.IGNORECASE):
                current_motion_mode = "G00"
            elif re.search(r"\bG0*1\b", line, re.IGNORECASE):
                current_motion_mode = "G01"
            elif re.search(r"\bG0*2\b", line, re.IGNORECASE):
                current_motion_mode = "G02"
            elif re.search(r"\bG0*3\b", line, re.IGNORECASE):
                current_motion_mode = "G03"

            match = re.search(r"^N(\d+)", line, re.IGNORECASE)
            if match:
                block_num = int(match.group(1))

                # Ekstraksi perubahan parameter sumbu
                x_match = re.search(r"\bX\s*=?\s*([-\d\.]+)", line, re.IGNORECASE)
                y_match = re.search(r"\bY\s*=?\s*([-\d\.]+)", line, re.IGNORECASE)
                z_match = re.search(r"\bZ\s*=?\s*([-\d\.]+)", line, re.IGNORECASE)
                b_match = re.search(r"\bB\s*=?\s*([-\d\.]+)", line, re.IGNORECASE)
                c_match = re.search(r"\bC\s*=?\s*([-\d\.]+)", line, re.IGNORECASE)
                a3_match = re.search(r"\bA3\s*=?\s*([-\d\.]+)", line, re.IGNORECASE)
                b3_match = re.search(r"\bB3\s*=?\s*([-\d\.]+)", line, re.IGNORECASE)
                c3_match = re.search(r"\bC3\s*=?\s*([-\d\.]+)", line, re.IGNORECASE)

                nx = float(x_match.group(1)) if x_match else x
                ny = float(y_match.group(1)) if y_match else y
                nz = float(z_match.group(1)) if z_match else z
                nb = float(b_match.group(1)) if b_match else b
                nc = float(c_match.group(1)) if c_match else c
                na3 = float(a3_match.group(1)) if a3_match else a3
                nb3 = float(b3_match.group(1)) if b3_match else b3
                nc3 = float(c3_match.group(1)) if c3_match else c3

                # Kalkulasi Delta Fisik Absolut
                dist_3d = np.sqrt((nx - x)**2 + (ny - y)**2 + (nz - z)**2)
                dist_rot = np.sqrt((nb - b)**2 + (nc - c)**2)
                dist_vec = np.sqrt((na3 - a3)**2 + (nb3 - b3)**2 + (nc3 - c3)**2)

                total_dist = dist_3d + dist_rot + dist_vec

                # Syarat 4: Cek keberadaan blok G01 yang tidak memproduksi pergerakan/delta (Phantom block)
                is_zero_g01 = (current_motion_mode == "G01") and (total_dist <= 1e-5)

                blocks.append({
                    'N': block_num,
                    'dist': total_dist,
                    'is_zero_g01': is_zero_g01
                })

                # Refresh koordinat memori
                x, y, z, b, c = nx, ny, nz, nb, nc
                a3, b3, c3 = na3, nb3, nc3

    return blocks

def extract_and_sync_data(csv_filepath, mpf_filepath):
    """Fungsi Integrasi SinuTrain dengan Interpolasi Jarak, Kolom Eksperimen, dan Feedrate Chase-back."""

    # 1. Tarik Peta G-Code Dasar
    mpf_blocks = get_mpf_blocks_with_distance(mpf_filepath)
    # Kamus khusus pencarian flag G01 secara super cepat
    mpf_block_flags = {b['N']: b['is_zero_g01'] for b in mpf_blocks}

    # 2. Bongkar Struktur CSV SinuTrain
    header_row = 0
    with open(csv_filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('time'):
                header_row = i
                break

    df = pd.read_csv(csv_filepath, skiprows=header_row)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    rename_dict = {}
    for col in df.columns:
        if 'time' in col: rename_dict[col] = 'time'
        elif 'f1' in col: rename_dict[col] = 'block_num'
        elif 'f7' in col: rename_dict[col] = 'feedrate'

    df = df.rename(columns=rename_dict)

    df['time'] = pd.to_numeric(df['time'])
    df['block_num'] = pd.to_numeric(df['block_num']).astype(int)
    df['feedrate'] = pd.to_numeric(df['feedrate'])

    interval = df['time'].diff().mode()[0]

    # --- SYARAT 1: Backward Fill Blok Negatif ke Blok G-Code Aktual di Bawahnya ---
    df['valid_block'] = df['block_num'].where(df['block_num'] > 0)
    df['valid_block'] = df['valid_block'].bfill().ffill()
    df['block_num'] = df['valid_block']

    df['group_id'] = (df['block_num'] != df['block_num'].shift(1)).cumsum()

    csv_results = []
    for group_id, group in df.groupby('group_id'):
        block = int(group['block_num'].iloc[0])
        duration = len(group) * interval

        # MENGGUNAKAN ARITHMETIC MEAN UNTUK DATA CONSTANT-TIME
        v_nonzero = group['feedrate'][group['feedrate'] > 0.001]
        if len(v_nonzero) > 0:
            mean_feedrate = np.mean(v_nonzero)
        else:
            mean_feedrate = 0.0

        csv_results.append({
            'Block_Number': block,
            'Time_Execution_s': duration,
            'Mean_Feedrate': mean_feedrate
        })

    # 3. SINKRONISASI ALGORITMA: Distance-Weighted Interpolation
    synced_results = []
    mpf_idx = 0
    csv_idx = 0

    while mpf_idx < len(mpf_blocks):
        if csv_idx >= len(csv_results):
            # Jika memori Trace SinuTrain sudah habis tapi G-Code masih sisa
            synced_results.append({
                'Block_Number': mpf_blocks[mpf_idx]['N'],
                'Time_Execution_s': 0.0,
                'Mean_Feedrate': 0.0,
                'Time_Execution_2_s': 0.0
            })
            mpf_idx += 1
            continue

        target_csv_b = csv_results[csv_idx]['Block_Number']

        # Penelusuran (Hunting) Sisa Blok G-Code yang sesuai dengan Log SinuTrain
        anchor_mpf_idx = -1
        for j in range(mpf_idx, len(mpf_blocks)):
            if mpf_blocks[j]['N'] == target_csv_b:
                anchor_mpf_idx = j
                break

        if anchor_mpf_idx == -1:
            # Jika blok SinuTrain tidak ditemukan sama sekali dalam sisa MPF (Blok anomali), buang.
            csv_idx += 1
            continue

        # Target Terkunci! Kumpulkan blok-blok yang hilang (jika ada) ke dalam 1 Kluster
        cluster = mpf_blocks[mpf_idx : anchor_mpf_idx + 1]

        anchor_csv = csv_results[csv_idx]
        anchor_time = anchor_csv['Time_Execution_s']
        anchor_feed = anchor_csv['Mean_Feedrate']

        total_dist = sum(b['dist'] for b in cluster)

        # --- SYARAT 2 & 3: Distribusi Proporsional Berdasarkan Jarak ---
        if total_dist > 1e-5:
            for block in cluster:
                weight = block['dist'] / total_dist
                block_time = anchor_time * weight

                # Syarat 3: Jika tidak ada pergerakan, dipaksa bernilai nol
                if block['dist'] > 1e-5:
                    block_feed = anchor_feed
                    # EKSPERIMEN: Perhitungan Time_Execution_2_s
                    # Rumus: Waktu(s) = (Jarak / Kecepatan Rata-rata) * 60 Detik
                    if block_feed > 0.0:
                        time_2 = (block['dist'] / block_feed) * 60.0
                    else:
                        time_2 = 0.0
                else:
                    block_feed = 0.0
                    time_2 = 0.0

                synced_results.append({
                    'Block_Number': block['N'],
                    'Time_Execution_s': round(block_time, 4),
                    'Mean_Feedrate': round(block_feed, 4),
                    'Time_Execution_2_s': round(time_2, 4)
                })
        else:
            # Jika semua baris pada kluster ini tidak bergerak murni, nol-kan semua
            # Kecuali jangkar (Anchor) agar durasinya aktualnya tetap terjaga (misal untuk Dwell / G04)
            for block in cluster:
                if block['N'] == target_csv_b:
                    synced_results.append({
                        'Block_Number': block['N'],
                        'Time_Execution_s': round(anchor_time, 4),
                        'Mean_Feedrate': round(anchor_feed, 4),
                        'Time_Execution_2_s': 0.0
                    })
                else:
                    synced_results.append({
                        'Block_Number': block['N'],
                        'Time_Execution_s': 0.0,
                        'Mean_Feedrate': 0.0,
                        'Time_Execution_2_s': 0.0
                    })

        # Majukan indeks ke blok selanjutnya
        mpf_idx = anchor_mpf_idx + 1
        csv_idx += 1

    # --- SYARAT 4: POST-PROCESSING (Penanganan G01 Tanpa Delta) ---
    # Digulirkan secara mundur (Reverse Iteration) agar Feedrate bisa meminjam (chase-back)
    # dari baris terdekat berikutnya meskipun saling berurutan
    for k in range(len(synced_results) - 1, -1, -1):
        b_num = synced_results[k]['Block_Number']

        # Periksa silang ID Baris dengan Flag Memori G01 dari Parser MPF di awal tadi
        if b_num in mpf_block_flags and mpf_block_flags[b_num]:
            # Bebaskan waktu eksekusinya menjadi murni nol
            synced_results[k]['Time_Execution_s'] = 0.0
            synced_results[k]['Time_Execution_2_s'] = 0.0

            # Duplikasi Mean Feedrate dari array setelahnya
            if k + 1 < len(synced_results):
                synced_results[k]['Mean_Feedrate'] = synced_results[k+1]['Mean_Feedrate']
            else:
                synced_results[k]['Mean_Feedrate'] = 0.0

    return pd.DataFrame(synced_results)


def batch_process_folder(input_folder, output_folder):
    """Memproses semua file CSV dan mencari pasangan MPF-nya di folder yang sama."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Membuat folder output baru: {output_folder}")

    search_pattern = os.path.join(input_folder, "*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"[WARNING] Tidak ditemukan file CSV di folder: {input_folder}")
        return

    print(f"[INFO] Ditemukan {len(csv_files)} file CSV. Memulai sinkronisasi batch fisika tinggi...")

    berhasil = 0
    gagal = 0

    for csv_filepath in csv_files:
        filename = os.path.basename(csv_filepath)
        base_name = os.path.splitext(filename)[0]

        # Cari file pasangan .mpf atau .nc
        mpf_filepath = os.path.join(input_folder, f"{base_name}.mpf")
        if not os.path.exists(mpf_filepath):
            mpf_filepath = os.path.join(input_folder, f"{base_name}.nc")

        if not os.path.exists(mpf_filepath):
            print(f"  [-] GAGAL: {filename}. (File G-Code .mpf/.nc pasangannya tidak ditemukan!)")
            gagal += 1
            continue

        output_filepath = os.path.join(output_folder, f"synced_{filename}")

        try:
            # Eksekusi fungsi sinkronisasi baru
            df_hasil = extract_and_sync_data(csv_filepath, mpf_filepath)
            df_hasil.to_csv(output_filepath, index=False)
            berhasil += 1
            print(f"  [+] Sukses tersinkronisasi: {filename} -> synced_{filename}")

        except Exception as e:
            gagal += 1
            print(f"  [-] GAGAL: {filename}. Error: {e}")

    print("-" * 50)
    print(f"[SELESAI] Proses Batch Selesai! Berhasil: {berhasil}, Gagal: {gagal}")

if __name__ == "__main__":
    # Ganti direktori ini sesuai konfigurasi lokal Anda jika diperlukan
    FOLDER_INPUT = "data_mentah_sinutrain"
    FOLDER_OUTPUT = "data_bersih_tersinkronisasi"
    batch_process_folder(FOLDER_INPUT, FOLDER_OUTPUT)
