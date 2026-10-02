import os
import glob
import pandas as pd
import numpy as np
import re

# ==========================================
# 1. KONFIGURASI FOLDER
# ==========================================
INPUT_FOLDER = './'         # Folder tempat file .mpf dan .csv berada
OUTPUT_FOLDER = './hasil'   # Folder untuk menyimpan file Excel hasil

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

mpf_files = glob.glob(os.path.join(INPUT_FOLDER, '*.mpf'))
print(f"Ditemukan {len(mpf_files)} file MPF. Memulai batch processing...\n")

# ==========================================
# 2. PROSES BATCH PER FILE
# ==========================================
for mpf_path in mpf_files:
    base_name = os.path.splitext(os.path.basename(mpf_path))[0]
    csv_path = os.path.join(INPUT_FOLDER, f"{base_name}.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️ Melewati '{base_name}': File CSV pasangannya tidak ditemukan.")
        continue

    print(f"⏳ Memproses: {base_name} ...")

    # ---------------------------------------
    # TAHAP A: Parsing Trace CSV (.csv)
    # ---------------------------------------
    with open(csv_path, 'r', encoding='latin-1') as f:
        lines = [f.readline() for _ in range(15)]
    header_idx = next(i for i, l in enumerate(lines) if l.startswith('time,'))

    df_trace = pd.read_csv(csv_path, skiprows=header_idx).dropna(subset=['time'])
    df_trace = df_trace.rename(columns={
        'f1\\s1': 'LineNum',
        'f2\\s2': 'X_act',
        'f3\\s3': 'Y_act',
        'f4\\s4': 'Z_act',
        'f7\\s7': 'V_act'
    })

    trace_times = df_trace['time'].to_numpy()
    trace_coords = df_trace[['X_act', 'Y_act', 'Z_act']].to_numpy()
    trace_velocs = df_trace['V_act'].to_numpy()
    trace_linenum = df_trace['LineNum'].to_numpy()

    # Ambil nomor blok valid yang terekam (abaikan nilai <= 0)
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

    # ---------------------------------------
    # TAHAP B: Parsing G-Code & Klasifikasi 3 Zona
    # ---------------------------------------
    nc_data = []
    current_xyz = [np.nan, np.nan, np.nan]
    current_segment = 1 # 1: Persiapan, 2: Cutting, 3: Postposition

    with open(mpf_path, 'r', encoding='latin-1') as f:
        for line in f:
            l = line.strip()

            # Deteksi transisi zona via comment tag
            if 'begin.txt' in l:
                current_segment = 2
            elif 'end.txt' in l:
                current_segment = 3

            clean = l.split(';')[0].strip()
            if not clean.startswith('N'): continue

            m = re.match(r'N(\d+)', clean)
            if not m: continue
            bid = int(m.group(1))

            x_m = re.search(r'X(-?\d+\.?\d*)', clean)
            y_m = re.search(r'Y(-?\d+\.?\d*)', clean)
            z_m = re.search(r'Z(-?\d+\.?\d*)', clean)

            if x_m: current_xyz[0] = float(x_m.group(1))
            if y_m: current_xyz[1] = float(y_m.group(1))
            if z_m: current_xyz[2] = float(z_m.group(1))

            nc_data.append({
                'Block_Num': bid,
                'Block_Text': f"N{bid}",
                'Raw': clean,
                'X': current_xyz[0],
                'Y': current_xyz[1],
                'Z': current_xyz[2],
                'Segment': current_segment
            })

    df_nc = pd.DataFrame(nc_data)
    if df_nc.empty: continue

    # Isi blok persiapan yang tanpa koordinat dengan nilai terdekat
    df_nc[['X', 'Y', 'Z']] = df_nc[['X', 'Y', 'Z']].ffill().bfill().fillna(0)
    df_nc['Dist'] = np.sqrt(df_nc['X'].diff()**2 + df_nc['Y'].diff()**2 + df_nc['Z'].diff()**2).fillna(0)

    # Pisahkan ke masing-masing kelompok data frame
    seg1_df = df_nc[df_nc['Segment'] == 1].copy()
    seg2_df = df_nc[df_nc['Segment'] == 2].copy()
    seg3_df = df_nc[df_nc['Segment'] == 3].copy()

    # ---------------------------------------
    # TAHAP C: Eksekusi Perhitungan Per Zona
    # ---------------------------------------
    matched_results = []

    # Fungsi bantu: Distribusi irisan indeks proporsional (Zona 1 & Zona 3)
    def process_proportional_zone(group_df, start_idx_bound, end_idx_bound, zone_name):
        results = []
        tot_samples = max(0, end_idx_bound - start_idx_bound)
        tot_dist = group_df['Dist'].sum()
        curr_slice_start = start_idx_bound

        for i in range(len(group_df)):
            row = group_df.iloc[i]
            if tot_dist > 0:
                porsi = row['Dist'] / tot_dist
            else:
                porsi = 1.0 / len(group_df)

            n_samples = int(round(porsi * tot_samples))
            curr_slice_end = min(curr_slice_start + n_samples, end_idx_bound)
            if i == len(group_df) - 1:
                curr_slice_end = end_idx_bound

            sl = df_trace.iloc[curr_slice_start:curr_slice_end]
            if len(sl) > 0:
                dur = sl['time'].iloc[-1] - sl['time'].iloc[0]
                v_avg = sl['V_act'].mean()
                v_max = sl['V_act'].max()
                safe_last_idx = min(curr_slice_end - 1, len(trace_coords) - 1)
                dev = np.linalg.norm(trace_coords[safe_last_idx] - np.array([row['X'], row['Y'], row['Z']]))
            else:
                dur, v_avg, v_max, dev = 0.0, 0.0, 0.0, 0.0

            est_dur = 0.0 if v_avg <= 0 else (row['Dist'] / (v_avg / 60.0))

            results.append({
                'Block Num': row['Block_Text'],
                'G-Code Raw': row['Raw'],
                'Zona Operasi': zone_name,
                'Jarak Tempuh (mm)': round(row['Dist'], 5),
                'Estimasi Durasi Teoritis (s)': round(est_dur, 5),
                'Durasi Trace Aktual (s)': round(dur, 5),
                'Feedrate Avg (mm/min)': round(v_avg, 2),
                'Feedrate Max (mm/min)': round(v_max, 2),
                'Penyimpangan Spasial (mm)': round(dev, 4)
            })
            curr_slice_start = curr_slice_end

        return results, curr_slice_start

    # --- 1. PROSES ZONA 1 (Persiapan) ---
    first_seg2_block = seg2_df['Block_Num'].iloc[0] if len(seg2_df) > 0 else 50
    ref_first_seg2 = get_next_valid_line(first_seg2_block)
    seg1_end_bound = np.where(trace_linenum == ref_first_seg2)[0][0] if ref_first_seg2 in valid_trace_lines else 100

    res_seg1, last_actual_idx = process_proportional_zone(seg1_df, 0, seg1_end_bound, '1 - Persiapan')
    matched_results.extend(res_seg1)

    # --- 2. PROSES ZONA 2 (Cutting Spasial Terkunci) ---
    for idx, row in seg2_df.iterrows():
        block_id = int(row['Block_Num'])
        target_xyz = np.array([row['X'], row['Y'], row['Z']])

        ref_block = get_next_valid_line(block_id)
        end_bound_idx = line_end_indices.get(ref_block, len(trace_coords) - 1)
        end_search_idx = min(end_bound_idx + 10, len(trace_coords))
        start_search_idx = min(last_actual_idx, end_search_idx)

        search_window = trace_coords[start_search_idx:end_search_idx]
        if len(search_window) == 0:
            actual_idx = start_search_idx
            safe_idx = min(actual_idx, len(trace_coords) - 1)
            min_dist = np.linalg.norm(trace_coords[safe_idx] - target_xyz)
        else:
            distances = np.linalg.norm(search_window - target_xyz, axis=1)
            min_local_idx = np.argmin(distances)
            actual_idx = start_search_idx + min_local_idx
            min_dist = distances[min_local_idx]

        duration = trace_times[actual_idx] - trace_times[last_actual_idx]
        if actual_idx > last_actual_idx:
            mean_feed = np.mean(trace_velocs[last_actual_idx:actual_idx])
            max_feed = np.max(trace_velocs[last_actual_idx:actual_idx])
        else:
            safe_idx = min(actual_idx, len(trace_velocs) - 1)
            mean_feed = trace_velocs[safe_idx]
            max_feed = trace_velocs[safe_idx]

        est_dur = 0.0 if mean_feed <= 0 else (row['Dist'] / (mean_feed / 60.0))

        matched_results.append({
            'Block Num': row['Block_Text'],
            'G-Code Raw': row['Raw'],
            'Zona Operasi': '2 - Cutting',
            'Jarak Tempuh (mm)': round(row['Dist'], 5),
            'Estimasi Durasi Teoritis (s)': round(est_dur, 5),
            'Durasi Trace Aktual (s)': round(duration, 5),
            'Feedrate Avg (mm/min)': round(mean_feed, 2),
            'Feedrate Max (mm/min)': round(max_feed, 2),
            'Penyimpangan Spasial (mm)': round(min_dist, 4)
        })
        last_actual_idx = actual_idx

    # --- 3. PROSES ZONA 3 (Postposition / Return Home) ---
    res_seg3, last_actual_idx = process_proportional_zone(seg3_df, last_actual_idx, len(trace_coords), '3 - Postposition')
    matched_results.extend(res_seg3)

    # ---------------------------------------
    # TAHAP D: Export Laporan Excel
    # ---------------------------------------
    df_hasil = pd.DataFrame(matched_results)
    out_filename = os.path.join(OUTPUT_FOLDER, f"{base_name}_Sinkronisasi_3Zona.xlsx")
    df_hasil.to_excel(out_filename, index=False)
    print(f"   ✅ Sukses! {len(df_hasil)} blok dipetakan. Disimpan ke: '{out_filename}'")

print("\n🎉 Semua proses sinkronisasi 3 zona selesai.")