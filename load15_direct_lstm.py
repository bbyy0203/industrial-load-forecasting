"""Direct-load seq2seq LSTM: (batch, 672, 5) + (batch, 96, 4) -> (batch, 96).

Predicts scaled load, not TSO residual. Import this module only after TensorFlow
is confirmed available.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from load15_direct_dataset import EXPECTED_HORIZON, EXPECTED_LOOKBACK


def build_direct_load_lstm(
    *,
    lookback: int = EXPECTED_LOOKBACK,
    horizon: int = EXPECTED_HORIZON,
    n_past_features: int = 5,
    n_future_features: int = 4,
    hidden: int = 32,
) -> keras.Model:
    if (lookback, horizon, n_past_features, n_future_features) != (
        EXPECTED_LOOKBACK, EXPECTED_HORIZON, 5, 4,
    ):
        raise ValueError(
            "当前第一版只接受接口 (672, 5) + (96, 4) -> (96,)。"
        )

    encoder_in = keras.Input(shape=(lookback, n_past_features), name="encoder_in")
    _, h, c = layers.LSTM(hidden, return_state=True, name="encoder_lstm")(encoder_in)

    decoder_in = keras.Input(shape=(horizon, n_future_features), name="decoder_in")
    decoder_seq = layers.LSTM(
        hidden,
        return_sequences=True,
        name="decoder_lstm",
    )(decoder_in, initial_state=[h, c])

    step_out = layers.TimeDistributed(
        layers.Dense(1, name="step_dense"),
        name="td_dense",
    )(decoder_seq)
    y = layers.Reshape((horizon,), name="load_scaled")(step_out)
    return keras.Model(inputs=[encoder_in, decoder_in], outputs=y, name="direct_load_lstm")


def compile_direct_load_lstm(model: keras.Model, *, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
