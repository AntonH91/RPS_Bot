from abc import ABC

from tensorflow.keras.layers import LSTM, Dense, InputLayer, BatchNormalization

import tensorflow as tf


class DefaultPredictionModel(tf.keras.Model):

    def get_config(self):
        super().get_config()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = InputLayer(input_shape=(None, 10))
        self.lstm1 = LSTM(units=16, return_sequences=True)
        self.lstm2 = LSTM(units=16)
        self.classifier = Dense(units=3, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        return self.classifier(x)

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, steps_per_execution=None, **kwargs):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.1,
                decay_steps=5,
                decay_rate=0.9
            ))
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution,
                        **kwargs)


class DefaultGameplayModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=64, activation='relu')

        self.output_layer = Dense(units=3, activation='softmax')

    def call(self, inputs, training=None, mask=None):

        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        return self.output_layer(x)

    def get_config(self):
        super().get_config()
