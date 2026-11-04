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
    def __init__(self, window_size: int = 301):
        self.window_size = window_size
        self.half_w = (window_size - 1) // 2  # 150 blocks

        # Inisialisasi Scaler
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.feature_cols = [
            'Cmd_F', 'Is_G01', 'Is_G02', 'Is_G03', 'Is_Traori',
            'Is_Cycle800', 'Is_MCALL_Sub', 'C832_Tol', 'C832_Mode',
            'Delta_X', 'Delta_Y', 'Delta_Z', 'Spline_Curvature',
            'Delta_3D', 'Delta_Rot', 'Tool_Vector_Delta', 'Kinematic_Blend_Ratio', 'Rotary_Velocity_Demand', 'Sharpness_Angle',
            'Is_Motion_Block', 'Is_Reversal_X', 'Is_Reversal_Y', 'Is_Reversal_Z', 'Turn_Count', 'Is_Partial_Arc', 'Is_Z_Plunge',
            'Kinematic_Speed_Limit', 'Delta_B', 'Delta_C', 'Delta_A3', 'Delta_B3', 'Delta_C3', 'Arc_Sweep'
        ]

    def _apply_log_transforms(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df_out = df.copy()

        # Kompresi logaritmik tambahan untuk seluruh sumbu putar dan vektor
        rot_features = ['Delta_Rot', 'Tool_Vector_Delta', 'Delta_B', 'Delta_C', 'Delta_A3', 'Delta_B3', 'Delta_C3', 'Arc_Sweep']
        for col in rot_features:
            if col in df_out.columns:
                df_out[col] = np.log1p(np.maximum(0.0, df_out[col].values))

        if 'Turn_Count' in df_out.columns:
            df_out['Turn_Count'] = np.log1p(np.maximum(0.0, df_out['Turn_Count'].values)) / np.log1p(100.0)

        # Kompresi logaritmik untuk meredam rentang ekstrem
        df_out['Delta_3D'] = np.log1p(np.maximum(0.0, df_out['Delta_3D'].values))
        if 'Kinematic_Speed_Limit' in df_out.columns:
            df_out['Kinematic_Speed_Limit'] = np.log1p(np.maximum(0.0, df_out['Kinematic_Speed_Limit'].values))
        df_out['Rotary_Velocity_Demand'] = np.log1p(np.maximum(0.0, df_out['Rotary_Velocity_Demand'].values))
        df_out['Cmd_F'] = np.log1p(np.maximum(0.0, df_out['Cmd_F'].values))
        df_out['Sharpness_Angle'] = df_out['Sharpness_Angle'].values / np.pi

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

    def fit_transform_dataset(self, df_list: List[pd.DataFrame], is_resume: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit scaler pada kumpulan data training dan kembalikan tensor (X, Y, W). (Mode High-RAM)"""
        if not is_resume:
            self.fit_scalers_only(df_list)

        X_all, Y_all, W_all = [], [], []
        for df in df_list:
            X_part, Y_part, W_part = self.transform_file(df, is_training=True)
            X_all.append(X_part)
            Y_all.append(Y_part)
            W_all.append(W_part)

        return np.concatenate(X_all, axis=0), np.concatenate(Y_all, axis=0), np.concatenate(W_all, axis=0)

    def get_padded_features(self, df_prep: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Hanya mengembalikan representasi 2D yang di-pad dan Y ter-scale untuk generator."""
        scaled_features = self.feature_scaler.transform(df_prep[self.feature_cols])

        binary_cols = ['Is_G01', 'Is_G02', 'Is_G03', 'Is_Traori', 'Is_Cycle800',
                       'Is_MCALL_Sub', 'Is_Motion_Block', 'Is_Reversal_X',
                       'Is_Reversal_Y', 'Is_Reversal_Z', 'Is_Partial_Arc', 'Is_Z_Plunge', 'Turn_Count']

        for col in binary_cols:
            if col in self.feature_cols:
                idx = self.feature_cols.index(col)
                scaled_features[:, idx] = df_prep[col].values

        standstill_df = df_prep.iloc[[0]].copy()

        if 'Cmd_F' in standstill_df.columns:
            standstill_df['Cmd_F'] = 0.0
        if 'Delta_X' in standstill_df.columns:
            standstill_df['Delta_X'] = 0.0
        if 'Delta_Y' in standstill_df.columns:
            standstill_df['Delta_Y'] = 0.0
        if 'Delta_Z' in standstill_df.columns:
            standstill_df['Delta_Z'] = 0.0
        if 'Spline_Curvature' in standstill_df.columns:
            standstill_df['Spline_Curvature'] = 0.0
        if 'Delta_3D' in standstill_df.columns:
            standstill_df['Delta_3D'] = 0.0
        if 'Delta_Rot' in standstill_df.columns:
            standstill_df['Delta_Rot'] = 0.0
        if 'Kinematic_Blend_Ratio' in standstill_df.columns:
            standstill_df['Kinematic_Blend_Ratio'] = 0.0
        if 'Rotary_Velocity_Demand' in standstill_df.columns:
            standstill_df['Rotary_Velocity_Demand'] = 0.0
        if 'Turn_Count' in standstill_df.columns:
            standstill_df['Turn_Count'] = 0.0
        if 'Is_Partial_Arc' in standstill_df.columns:
            standstill_df['Is_Partial_Arc'] = 0
        if 'Is_Z_Plunge' in standstill_df.columns:
            standstill_df['Is_Z_Plunge'] = 0
        if 'Kinematic_Speed_Limit' in standstill_df.columns:
            standstill_df['Kinematic_Speed_Limit'] = 0.0
        for col in ['Delta_B', 'Delta_C', 'Delta_A3', 'Delta_B3', 'Delta_C3', 'Arc_Sweep']:
            if col in standstill_df.columns:
                standstill_df[col] = 0.0
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

        # Buat bobot sampel: Penalti 5x untuk Helical Drilling, 3x HANYA JIKA TRAORI aktif DAN ada pergerakan rotari, lainnya 1.0
        is_moving_rotary = (df_prep['Delta_Rot'].values > 1e-4) | (df_prep['Tool_Vector_Delta'].values > 1e-4)
        is_simultaneous = (df_prep['Is_Traori'].values == 1) & is_moving_rotary
        is_helical = df_prep['Turn_Count'].values > 0
        sample_weights = np.where(is_helical, 5.0, np.where(is_simultaneous, 3.0, 1.0)).astype(np.float32)

        return padded_features, scaled_target, sample_weights

    def transform_file(self, df: pd.DataFrame, is_training: bool = True):
        df_prep = self._apply_log_transforms(df, is_training=is_training)

        padded_features, scaled_target, sample_weights = self.get_padded_features(df_prep)

        num_samples = len(df_prep)
        num_features = len(self.feature_cols)
        X_windows = np.zeros((num_samples, self.window_size, num_features), dtype=np.float32)

        for i in range(num_samples):
            X_windows[i] = padded_features[i : i + self.window_size]

        if is_training:
            return X_windows, scaled_target, sample_weights

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
    Generator Data Low-RAM berbasis Stateless NumPy Indexing (Thread-Safe untuk Keras 3 Multi-Workers).
    Hanya menyimpan matriks 2D di RAM dan memetakan indeks global (file_idx, row_idx) secara langsung
    tanpa antrean buffer yang rawan kehilangan sampel atau race-condition.
    """
    def __init__(self, df_list: List[pd.DataFrame], preprocessor: DatasetPreprocessor,
                 batch_size: int = 128, is_training: bool = True, shuffle: bool = None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.window_size = preprocessor.window_size
        self.is_training = is_training
        self.shuffle = is_training if shuffle is None else shuffle
        self.num_features = len(preprocessor.feature_cols)

        self.X_padded_list = []
        self.Y_list = []
        self.W_list = []
        file_idx_chunks = []
        row_idx_chunks = []

        for file_idx, df in enumerate(df_list):
            df_prep = preprocessor._apply_log_transforms(df, is_training=self.is_training)
            padded_feat, scaled_y, sample_w = preprocessor.get_padded_features(df_prep)

            self.X_padded_list.append(np.ascontiguousarray(padded_feat, dtype=np.float32))
            if self.is_training and scaled_y is not None:
                self.Y_list.append(np.ascontiguousarray(scaled_y, dtype=np.float32))
                self.W_list.append(np.ascontiguousarray(sample_w, dtype=np.float32))

            num_rows = len(df)
            file_idx_chunks.append(np.full(num_rows, file_idx, dtype=np.int32))
            row_idx_chunks.append(np.arange(num_rows, dtype=np.int32))

        if file_idx_chunks:
            self.file_idx_arr = np.concatenate(file_idx_chunks)
            self.row_idx_arr = np.concatenate(row_idx_chunks)
        else:
            self.file_idx_arr = np.array([], dtype=np.int32)
            self.row_idx_arr = np.array([], dtype=np.int32)

        self.total_samples = len(self.file_idx_arr)
        self.batches_per_epoch = int(np.ceil(self.total_samples / float(self.batch_size)))
        self.indices = np.arange(self.total_samples, dtype=np.int32)

        self.on_epoch_end()

    def on_epoch_end(self):
        """Mengacak urutan indeks global setiap akhir epoch (hanya jika shuffle=True)."""
        if self.shuffle and self.total_samples > 0:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.total_samples)
        batch_indices = self.indices[start_idx:end_idx]
        count = len(batch_indices)

        batch_x = np.empty((count, self.window_size, self.num_features), dtype=np.float32)
        f_arr = self.file_idx_arr[batch_indices]
        r_arr = self.row_idx_arr[batch_indices]

        for i in range(count):
            f_i = f_arr[i]
            r_i = r_arr[i]
            batch_x[i] = self.X_padded_list[f_i][r_i : r_i + self.window_size]

        if not self.is_training:
            return batch_x

        batch_y = np.empty((count, 1), dtype=np.float32)
        batch_w = np.empty((count,), dtype=np.float32)
        for i in range(count):
            f_i = f_arr[i]
            r_i = r_arr[i]
            batch_y[i] = self.Y_list[f_i][r_i]
            batch_w[i] = self.W_list[f_i][r_i]

        return batch_x, batch_y, batch_w
