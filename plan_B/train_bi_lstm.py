"""
train_bi_lstm.py
Tahap 4: Pelatihan Model Dual-Layer Bi-LSTM untuk Prediksi Profil Kecepatan CNC.
"""

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
    # Mengambil indeks tengah secara otomatis, W=301 -> Index 150
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
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0), metrics=["mae", "mse"])

    return model

import os

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

    training_callbacks = [
        callbacks.ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
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
        history = model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=training_callbacks,
            verbose=1
        )

    return model, history
