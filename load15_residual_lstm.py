"""Comparable encoder-decoder LSTM for last-week residual correction."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_residual_lstm(hidden: int = 32) -> keras.Model:
    encoder_in = keras.Input(shape=(672, 5), name="encoder_in")
    _, h, c = layers.LSTM(hidden, return_state=True, name="encoder_lstm")(encoder_in)
    decoder_in = keras.Input(shape=(96, 6), name="decoder_in")
    decoder = layers.LSTM(hidden, return_sequences=True, name="decoder_lstm")(
        decoder_in, initial_state=[h, c]
    )
    steps = layers.TimeDistributed(layers.Dense(1, name="step_dense"), name="td_dense")(decoder)
    output = layers.Reshape((96,), name="residual_scaled")(steps)
    return keras.Model([encoder_in, decoder_in], output, name="last_week_residual_lstm")


def compile_residual_lstm(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
