import random

from rps.constants import Move, GameState
from rps.agent.base_agent import RPSAgent
from rps.utils.utils import get_categorical_move
from collections import deque

import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


class AiAgent(RPSAgent):
    moves = [Move.ROCK, Move.PAPER, Move.SCISSORS]

    def __init__(self, prediction_model: tf.keras.Model, gameplay_model: tf.keras.models.Model,
                 move_batch_size=16, training=True,
                 rewards={GameState.WIN: 1, GameState.DRAW: 0, GameState.LOSS: -1}) -> None:
        super().__init__()

        self.history = []
        self.prediction_history = []

        self.prediction_model = prediction_model
        self.initial_prediction_weights = None

        self.gameplay_model = gameplay_model

        self.training = training
        self.rewards = rewards
        self.move_batch_size = move_batch_size

        self.last_estimated_move = None

        self.estimation_hits = 0.0
        self.estimation_misses = 0.0
        self.played_rounds = 0

        self.last_observations = None

    def reset(self):
        super().reset()

        self.last_estimated_move = None
        self.history = []
        self.prediction_history = []
        self.estimation_hits = 0.0
        self.estimation_misses = 0.0
        self.played_rounds = 0
        self.last_observations = None

        # Reset weights for the prediction model, so it can learn about a new agent
        if not self.initial_prediction_weights is None:
            self.prediction_model.set_weights(self.initial_prediction_weights)

    def play(self, opponent_move: Move) -> Move:
        return self.get_next_move(self.last_observations)

    def game_state(self, own_move: Move, opponent_move: Move, result: GameState):
        super().game_state(own_move, opponent_move, result)

        self.history.append([own_move, opponent_move, result, self.last_estimated_move, self.hit_ratio()])

        if self.last_estimated_move == opponent_move:
            self.estimation_hits += 1
        else:
            self.estimation_misses += 1

        # Update the prediction model weights
        if self.played_rounds > self.move_batch_size and self.played_rounds % self.move_batch_size == 0:
            self.update_prediction_weights()

        # Estimate a new opponent move
        next_opponent_move = self.estimate_opponent_move()
        self.last_estimated_move = next_opponent_move

        # Record the latest set of observations for the next round of the game
        self.last_observations = [own_move, opponent_move, result, next_opponent_move, self.hit_ratio()]

        self.played_rounds += 1

    def update_prediction_weights(self):
        """Updates the weights on the predictive model."""
        # Get twice the batch data, so that there is a batch_size set of data and labels
        batch_size = self.move_batch_size

        history = self.get_history_observations(batch_size * 2)
        ts_data = history[0, :, :]

        inputs = np.empty((batch_size, batch_size, 10))
        labels = np.empty((batch_size, 3))
        for index in range(batch_size):
            inputs[index] = ts_data[index: batch_size + index, :]
            labels[index] = ts_data[index + batch_size, len(Move):len(Move) * 2]

        self.prediction_model.fit(inputs, labels)

    def estimate_opponent_move(self) -> Move:
        """Runs the predictive model to estimate the opponents' next move"""
        obs = self.get_history_observations(self.move_batch_size)
        if obs is not None:
            result = self.prediction_model.predict(obs)
            result = np.argmax(result)

            if self.initial_prediction_weights is None:
                self.initial_prediction_weights = self.prediction_model.get_weights()
        else:
            result = random.randint(0, len(self.moves) - 1)
        return self.moves[result]

    def get_next_move(self, observations):
        """Runs the gameplay model to get the next move.

        @:param observations Array containing [own_last_move, opponent_last_move, outcome, predicted_move, hit_ratio]
        @:return Move enum value corresponding to the next intended Move"""

        if observations is not None:
            own_last_move, opponent_last_move, outcome, predicted_move, hit_ratio = observations

            # Compute the categorical of the observation
            own_move = get_categorical_move(own_last_move)
            opposing_move = get_categorical_move(opponent_last_move)
            result = to_categorical(outcome.value, len(GameState))
            predicted_move = get_categorical_move(predicted_move)

            # Concatenate it all to a numpy array for feeding into the network
            inputs = np.concatenate([own_move, opposing_move, result, predicted_move, [hit_ratio]])
            inputs = inputs.reshape((1,) + inputs.shape)
            # Add the next prediction history
            self.prediction_history.append(np.concatenate([own_move, opposing_move, result]))

            # Predict the next move
            next_move = self.gameplay_model.predict(inputs)
            next_move = self.moves[np.argmax(next_move)]
        else:
            next_move = random.choice(self.moves)
        return next_move

    def hit_ratio(self):
        """Calculates how many times the predictive model has been correct, for observation purposes"""
        if self.estimation_misses + self.estimation_hits == 0:
            return 0.0
        else:
            return self.estimation_hits / (self.estimation_misses + self.estimation_hits)

    def get_history_observations(self, number):
        if len(self.prediction_history) > 0:
            observations = self.prediction_history[-number:]
            observations = np.vstack(observations)
            # Reshape to add batch dimension
            observations = observations.reshape((1,) + observations.shape)
            return observations
