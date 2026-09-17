"""
train_bi_lstm.py
Tahap 4: Pelatihan Model Dual-Layer Bi-LSTM untuk Prediksi Profil Kecepatan CNC.
"""

import keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import numpy as np
from typing import Tuple

import tensorflow.keras.backend as K

def build_bilstm_model(input_shape: Tuple[int, int], learning_rate: float = 1e-3, lstm_units: int = 256) -> tf.keras.Model:
    """Membangun arsitektur Dual-Layer Bi-LSTM dengan Multi-Head Attention dan Dinamis Center Bypass."""
    inputs = layers.Input(shape=input_shape, name="NC_Sequence_Input")

    # 1. Jalur Utama (LSTM Navigator)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, name="forward_BiLSTM_L1"))(inputs)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, name="forward_BiLSTM_L2"))(x)
    x = layers.LayerNormalization(name="Sequence_Layer_Norm")(x)

    # Mengganti Kacamata: Multi-Head Attention Mechanism
    attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=64, name="Multi_Head_Attention")(x, x)
    pooled_x = layers.GlobalAveragePooling1D(name="Attention_Pooling")(attention_out)

    # 2. Jalur Pintas Dinamis (Center-Block Bypass)
    # Mengambil indeks tengah secara otomatis
    center_idx = input_shape[0] // 2
    center_block = layers.Lambda(lambda tensor: tensor[:, center_idx, :], name="Center_Block_Features")(inputs)

    # 3. Penggabungan (Concatenate)
    merged = layers.Concatenate(name="Attention_and_Bypass_Concat")([pooled_x, center_block])

    # 4. Dense Regressor Head
    d = layers.Dense(256, activation="relu")(merged)
    d = layers.Dropout(0.3)(d)
    d = layers.Dense(64, activation="relu")(d)
    outputs = layers.Dense(1, activation="linear", name="Normalized_Feedrate_Output")(d)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNC_Kinematics_BiLSTM_Plan_B")
    optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=0.1), metrics=["mae", "mse"], jit_compile=True)

    return model

import os

class DualMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.5, patience_lr=3, patience_stop=10, min_lr=1e-6):
        super(DualMonitorCallback, self).__init__()
        self.factor = factor
        self.patience_lr = patience_lr
        self.patience_stop = patience_stop
        self.min_lr = min_lr

        self.wait_lr = 0
        self.wait_stop = 0
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        current_val_mae = logs.get('val_mae')

        if current_val_loss is None or current_val_mae is None:
            return

        improved_loss = current_val_loss < (self.best_val_loss - 1e-5)
        improved_mae = current_val_mae < (self.best_val_mae - 1e-5)

        any_improvement = improved_loss or improved_mae

        if any_improvement:
            if improved_loss:
                self.best_val_loss = current_val_loss
            if improved_mae:
                self.best_val_mae = current_val_mae

            self.wait_lr = 0
            self.wait_stop = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait_lr += 1
            self.wait_stop += 1

            if self.wait_lr >= self.patience_lr:
                # Keras 3 / TF 2 safe LR reduction
                if hasattr(self.model.optimizer, 'learning_rate'):
                    lr_attr = 'learning_rate'
                elif hasattr(self.model.optimizer, 'lr'):
                    lr_attr = 'lr'
                else:
                    raise AttributeError("Optimizer does not have 'learning_rate' or 'lr' attribute.")

                lr_var = getattr(self.model.optimizer, lr_attr)

                try:
                    if hasattr(lr_var, 'numpy'):
                        old_lr = float(lr_var.numpy())
                    else:
                        old_lr = float(K.get_value(lr_var))
                except Exception:
                    old_lr = float(lr_var)

                new_lr = max(old_lr * self.factor, self.min_lr)

                if old_lr > self.min_lr:
                    try:
                        if hasattr(lr_var, 'assign'):
                            lr_var.assign(new_lr)
                        else:
                            K.set_value(lr_var, new_lr)
                    except Exception:
                        # Fallback for Keras 3 direct assignment
                        setattr(self.model.optimizer, lr_attr, new_lr)

                    print(f"\n[DualMonitor] Epoch {epoch+1}: Keduanya stagnan. Menurunkan learning rate menjadi {new_lr}.")
                self.wait_lr = 0

            if self.wait_stop >= self.patience_stop:
                print(f"\n[DualMonitor] Epoch {epoch+1}: Early Stopping aktif! val_loss dan val_mae tidak turun selama {self.patience_stop} epoch.")
                self.model.stop_training = True
                if self.best_weights is not None:
                    print("[DualMonitor] Mengembalikan bobot model ke kondisi terbaik (Restore Best Weights).")
                    self.model.set_weights(self.best_weights)

def run_training(train_data, val_data,
                 input_shape: Tuple[int, int],
                 model_save_path: str = "bilstm_feedrate_model.keras",
                 checkpoint_dir: str = None,
                 resume_model_path: str = None,
                 learning_rate: float = 1e-3,
                 initial_epoch: int = 0,
                 lstm_units: int = 256,
                 epochs: int = 200):
    """
    Menjalankan pelatihan.
    train_data dan val_data bisa berupa tuple (X, Y) untuk mode numpy biasa,
    atau berupa tf.keras.utils.Sequence / generator untuk mode low-RAM.
    Jika resume_model_path diberikan, maka lanjutkan pelatihan dari model tersebut.
    """
    model = build_bilstm_model(input_shape=input_shape, learning_rate=learning_rate, lstm_units=lstm_units)

    if resume_model_path and os.path.exists(resume_model_path):
        print(f"[INFO] Meresume (Transfer Learning) dari bobot model: {resume_model_path}")
        # Menghindari bug deserialisasi GlorotUniform di Keras 3 dengan hanya me-load bobotnya ke kerangka baru
        model.load_weights(resume_model_path)

    model.summary()

    log_dir = os.path.dirname(model_save_path)
    if not log_dir:
        log_dir = "."
    csv_log_path = os.path.join(log_dir, "training_history_log.csv")

    training_callbacks = [
        callbacks.ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1),
        DualMonitorCallback(factor=0.5, patience_lr=3, patience_stop=10, min_lr=1e-6),
        callbacks.CSVLogger(csv_log_path, separator=",", append=True)
    ]

    # Tambahkan autosave (overwrite) setiap epoch ke dalam folder jika diminta
    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # Menggunakan nama statis untuk memaksa Overwrite per epoch
        epoch_save_path = os.path.join(checkpoint_dir, "latest_model_checkpoint.keras")
        training_callbacks.append(
            callbacks.ModelCheckpoint(epoch_save_path, save_best_only=False, verbose=0)
        )

    if isinstance(train_data, tuple):
        # Mode High RAM (numpy arrays)
        X_train, Y_train = train_data
        X_val, Y_val = val_data
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=128,  # Batch size ini untuk tensorflow model fit jika pakai mode numpy array (HIGH RAM). Karena Low RAM pake generator, batch size diatur di generatornya.
            callbacks=training_callbacks,
            verbose=1
        )
    else:
        # Mode Low RAM (generator)
        fit_kwargs = {
            "x": train_data,
            "validation_data": val_data,
            "epochs": epochs,
            "initial_epoch": initial_epoch,
            "callbacks": training_callbacks,
            "verbose": 1
        }

        # Deteksi otomatis: Gunakan Multiprocessing eksplisit HANYA jika menggunakan Keras 2
        if int(keras.__version__.split('.')[0]) < 3:
            fit_kwargs["workers"] = 8
            fit_kwargs["use_multiprocessing"] = True
            fit_kwargs["max_queue_size"] = 20
            print("\n[INFO] Keras 2 terdeteksi. Akselerasi Multiprocessing CPU (8 Workers) AKTIF!")
        else:
            print("\n[INFO] Keras 3 terdeteksi. Menggunakan sistem Data API bawaan.")

        history = model.fit(**fit_kwargs)

    return model, history
