from abc import ABC

from tensorflow.keras.layers import LSTM, Dense, InputLayer, BatchNormalization

import tensorflow as tf


class DefaultPredictionModel(tf.keras.Model):

    def get_config(self):
        super().get_config()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = InputLayer(input_shape=(None, 10))

        self.lstms = [LSTM(units=32, return_sequences=True) for _ in range(10)]

        self.lstm_out = LSTM(units=32, return_sequences=False)
        self.classifier = Dense(units=3, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)

        for lstm in self.lstms:
            x = lstm(x)

        x = self.lstm_out(x)
        return self.classifier(x)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, steps_per_execution=None, **kwargs):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.1,
                decay_steps=2,
                decay_rate=0.6
            ))
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution,
                        **kwargs)


class DefaultGameplayModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = InputLayer(input_shape=(1, 14))

        self.dense_layers = [Dense(units=256, activation='relu') for _ in range(12)]

        self.output_layer = Dense(units=3, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)

        for dense in self.dense_layers:
            x = dense(x)

        return self.output_layer(x)

    def get_config(self):
        super().get_config()
