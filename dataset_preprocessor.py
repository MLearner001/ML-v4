"""
dataset_preprocessor.py
Tahap 3: Penskalaan Fitur Hybrid, Boundary Padding, dan Sliding Window Generator W=201.
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict

class DatasetPreprocessor:
    def __init__(self, window_size: int = 101):
        self.window_size = window_size
        self.half_w = (window_size - 1) // 2  # 50 blocks

        # Inisialisasi Scaler
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.feature_cols = [
            'Cmd_F', 'Cmd_S', 'Is_G01', 'Is_G02', 'Is_G03', 'Is_Traori',
            'Is_Cycle800', 'Is_MCALL_Sub', 'C832_Tol', 'C832_Mode',
            'Delta_3D', 'Delta_Rot', 'Tool_Vector_Delta', 'Kinematic_Blend_Ratio', 'Rotary_Velocity_Demand', 'Sharpness_Angle',
            'Is_Motion_Block', 'Is_Reversal_X', 'Is_Reversal_Y', 'Is_Reversal_Z', 'Theo_Duration', 'Turn_Count', 'Is_Partial_Arc'
        ]

    def _apply_log_transforms(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df_out = df.copy()

        # Kompresi logaritmik untuk meredam rentang ekstrem
        df_out['Delta_3D'] = np.log1p(np.maximum(0.0, df_out['Delta_3D'].values))
        df_out['Rotary_Velocity_Demand'] = np.log1p(np.maximum(0.0, df_out['Rotary_Velocity_Demand'].values))
        df_out['Cmd_F'] = np.log1p(np.maximum(0.0, df_out['Cmd_F'].values))
        df_out['Sharpness_Angle'] = df_out['Sharpness_Angle'].values / np.pi

        # Log kompresi untuk Theo_Duration karena rentangnya bisa bervariasi dari ms hingga menit
        if 'Theo_Duration' in df_out.columns:
            df_out['Theo_Duration'] = np.log1p(np.maximum(0.0, df_out['Theo_Duration'].values))

        # Log kompresi untuk blend ratio karena bisa meledak saat translasi = 0
        if 'Kinematic_Blend_Ratio' in df_out.columns:
            df_out['Kinematic_Blend_Ratio'] = np.log1p(np.maximum(0.0, df_out['Kinematic_Blend_Ratio'].values))

        if is_training and 'Target_Feedrate' in df_out.columns:
            df_out['Target_Feedrate'] = np.log1p(np.maximum(0.0, df_out['Target_Feedrate'].values))

        return df_out

    def fit_scalers_only(self, df_list: List[pd.DataFrame]):
        """Fit scaler pada kumpulan data tanpa membuat array 3D di memori."""
        # Menggunakan loop parsial atau concat (concat masih aman untuk memori 2D)
        combined_df = pd.concat([self._apply_log_transforms(df, is_training=True) for df in df_list], axis=0)
        self.feature_scaler.fit(combined_df[self.feature_cols])
        self.target_scaler.fit(combined_df[['Target_Feedrate']])

    def fit_transform_dataset(self, df_list: List[pd.DataFrame], is_resume: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler pada kumpulan data training dan kembalikan tensor (X, Y). (Mode High-RAM)"""
        if not is_resume:
            self.fit_scalers_only(df_list)

        X_all, Y_all = [], []
        for df in df_list:
            X_part, Y_part = self.transform_file(df, is_training=True)
            X_all.append(X_part)
            Y_all.append(Y_part)

        return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0)

    def get_padded_features(self, df_prep: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Hanya mengembalikan representasi 2D yang di-pad dan Y ter-scale untuk generator."""
        scaled_features = self.feature_scaler.transform(df_prep[self.feature_cols])

        binary_cols = ['Is_G01', 'Is_G02', 'Is_G03', 'Is_Traori', 'Is_Cycle800',
                       'Is_MCALL_Sub', 'Is_Motion_Block', 'Is_Reversal_X',
                       'Is_Reversal_Y', 'Is_Reversal_Z', 'Is_Partial_Arc']

        for col in binary_cols:
            if col in self.feature_cols:
                idx = self.feature_cols.index(col)
                scaled_features[:, idx] = df_prep[col].values

        standstill_df = df_prep.iloc[[0]].copy()

        if 'Cmd_F' in standstill_df.columns:
            standstill_df['Cmd_F'] = 0.0
        if 'Delta_3D' in standstill_df.columns:
            standstill_df['Delta_3D'] = 0.0
        if 'Delta_Rot' in standstill_df.columns:
            standstill_df['Delta_Rot'] = 0.0
        if 'Kinematic_Blend_Ratio' in standstill_df.columns:
            standstill_df['Kinematic_Blend_Ratio'] = 0.0
        if 'Rotary_Velocity_Demand' in standstill_df.columns:
            standstill_df['Rotary_Velocity_Demand'] = 0.0
        if 'Theo_Duration' in standstill_df.columns:
            # Durasi diam = 0
            standstill_df['Theo_Duration'] = 0.0
        if 'Turn_Count' in standstill_df.columns:
            standstill_df['Turn_Count'] = 0.0
        if 'Is_Partial_Arc' in standstill_df.columns:
            standstill_df['Is_Partial_Arc'] = 0
        if 'Target_Feedrate' in standstill_df.columns:
            standstill_df['Target_Feedrate'] = 0.0

        scaled_standstill = self.feature_scaler.transform(standstill_df[self.feature_cols])

        for col in binary_cols:
            if col in self.feature_cols:
                idx = self.feature_cols.index(col)
                if col in ['Is_Motion_Block', 'Is_G01', 'Is_G02', 'Is_G03']:
                    scaled_standstill[0, idx] = 0
                else:
                    scaled_standstill[0, idx] = standstill_df[col].values[0]

        padded_features = np.vstack([
            np.repeat(scaled_standstill, self.half_w, axis=0),
            scaled_features,
            np.repeat(scaled_standstill, self.half_w, axis=0)
        ])

        scaled_target = self.target_scaler.transform(df_prep[['Target_Feedrate']]).astype(np.float32) if 'Target_Feedrate' in df_prep.columns else None

        return padded_features, scaled_target

    def transform_file(self, df: pd.DataFrame, is_training: bool = True):
        df_prep = self._apply_log_transforms(df, is_training=is_training)

        padded_features, scaled_target = self.get_padded_features(df_prep)

        num_samples = len(df_prep)
        num_features = len(self.feature_cols)
        X_windows = np.zeros((num_samples, self.window_size, num_features), dtype=np.float32)

        for i in range(num_samples):
            X_windows[i] = padded_features[i : i + self.window_size]

        if is_training:
            return X_windows, scaled_target

        return X_windows

    def save_scalers(self, path: str = "scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "feature_cols": self.feature_cols
            }, f)

    def load_scalers(self, path: str = "scaler.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.feature_scaler = data["feature_scaler"]
            self.target_scaler = data["target_scaler"]
            self.feature_cols = data["feature_cols"]

class SlidingWindowGenerator(tf.keras.utils.Sequence):
    """
    Generator Data Keras dengan mekanisme Interleaved Memory Buffer.
    Mencegah OOM dengan tidak menaruh array 3D di memori sekaligus,
    dan mengatasi Correlated Batches dengan membaca beberapa file secara paralel
    ke dalam buffer lalu diacak sebelum disuplai ke GPU.
    """
    def __init__(self, df_list: List[pd.DataFrame], preprocessor: DatasetPreprocessor, batch_size: int = 128, is_training: bool = True, max_buffer_size: int = 5000, parallel_files: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.window_size = preprocessor.window_size
        self.is_training = is_training

        self.max_buffer_size = max_buffer_size
        self.parallel_files = parallel_files

        # Pre-compute padded 2D features untuk mempercepat ekstraksi window
        self.X_padded_list = []
        self.Y_list = []
        self.file_lengths = []

        self.total_samples = 0

        # Mapping index global ke lokasi lokal untuk fallback mode stateless (inferensi)
        self.file_indices = []

        for file_idx, df in enumerate(df_list):
            df_prep = preprocessor._apply_log_transforms(df, is_training=self.is_training)
            padded_feat, scaled_y = preprocessor.get_padded_features(df_prep)
            self.X_padded_list.append(padded_feat)
            if self.is_training:
                self.Y_list.append(scaled_y)

            num_rows = len(df)
            self.file_lengths.append(num_rows)
            self.total_samples += num_rows

            for r in range(num_rows):
                self.file_indices.append((file_idx, r))

        self.num_files = len(df_list)
        self.batches_per_epoch = int(np.ceil(self.total_samples / float(self.batch_size)))

        if self.is_training:
            self.on_epoch_end()

    def on_epoch_end(self):
        """Reset pointer pembacaan setiap awal epoch (hanya berlaku jika training)."""
        if not self.is_training:
            return

        self.file_order = np.random.permutation(self.num_files)
        self.active_files = [] # List of tuples: [file_idx, current_row]
        self.next_file_idx = 0

        # Load initial active files
        self._replenish_active_files()

        self.buffer_x = []
        self.buffer_y = []

    def _replenish_active_files(self):
        """Menambah file aktif hingga mencapai batas parallel_files"""
        while len(self.active_files) < self.parallel_files and self.next_file_idx < self.num_files:
            f_idx = self.file_order[self.next_file_idx]
            self.active_files.append([f_idx, 0]) # [file_index, current_row_pointer]
            self.next_file_idx += 1

    def _fill_buffer(self):
        """Membaca window secara sekuensial (bergiliran/interleaved) dari file-file aktif hingga buffer penuh."""
        while len(self.buffer_x) < self.max_buffer_size and len(self.active_files) > 0:
            for active_f in self.active_files.copy():
                f_idx, r_idx = active_f

                # Baca 1 window
                window = self.X_padded_list[f_idx][r_idx : r_idx + self.window_size]
                self.buffer_x.append(window)
                if self.is_training:
                    self.buffer_y.append(self.Y_list[f_idx][r_idx])

                # Majukan pointer
                active_f[1] += 1

                # Jika file sudah habis terbaca
                if active_f[1] >= self.file_lengths[f_idx]:
                    self.active_files.remove(active_f)
                    self._replenish_active_files()

                if len(self.buffer_x) >= self.max_buffer_size:
                    break

        # Shuffle murni di dalam RAM (Buffer) jika sedang training
        if self.is_training and len(self.buffer_x) > 0:
            indices = np.arange(len(self.buffer_x))
            np.random.shuffle(indices)
            self.buffer_x = [self.buffer_x[i] for i in indices]
            self.buffer_y = [self.buffer_y[i] for i in indices]

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        if not self.is_training:
            # Mode Stateless murni untuk inferensi agar validasi dimensi atau `predict` Keras
            # tidak merusak urutan antrean stateful buffer (mencegah shape mismatch error).
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.total_samples)

            batch_x = []
            batch_y = []

            for i in range(start_idx, end_idx):
                file_idx, row_idx = self.file_indices[i]
                padded_arr = self.X_padded_list[file_idx]
                window = padded_arr[row_idx : row_idx + self.window_size]
                batch_x.append(window)

            return np.array(batch_x, dtype=np.float32)

        # Mode Stateful Interleaved Buffer (Training)
        if len(self.buffer_x) < self.batch_size and (len(self.active_files) > 0 or self.next_file_idx < self.num_files):
            self._fill_buffer()

        # Ambil batch_size item dari buffer
        take_count = min(self.batch_size, len(self.buffer_x))
        if take_count == 0:
            # Fallback jika kehabisan data di akhir epoch
            return (np.zeros((1, self.window_size, len(self.preprocessor.feature_cols))), np.zeros((1, 1)))

        batch_x = np.array(self.buffer_x[:take_count], dtype=np.float32)
        batch_y = np.array(self.buffer_y[:take_count], dtype=np.float32)

        # Hapus item yang sudah diambil dari buffer
        self.buffer_x = self.buffer_x[take_count:]
        self.buffer_y = self.buffer_y[take_count:]

        return batch_x, batch_y