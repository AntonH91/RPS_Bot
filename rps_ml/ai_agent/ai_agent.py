import random
import csv
from rps.constants import Move, GameState
from rps.agent.base_agent import RPSAgent
from rps.utils.utils import get_categorical_move
from collections import deque

import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from pathlib import Path


class AiAgent(RPSAgent):
    moves = [Move.ROCK, Move.PAPER, Move.SCISSORS]

    def __init__(self, prediction_model: tf.keras.Model, gameplay_model: tf.keras.models.Model,
                 move_batch_size=16, training=True,
                 rewards=None, epsilon=0.2, epsilon_decay=None) -> None:
        super().__init__()

        if rewards is None:
            rewards = {GameState.WIN: 1, GameState.DRAW: 0, GameState.LOSS: -1, GameState.DISQUALIFIED: -10}

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
        self.state = None
        self.last_observations = None

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Should be loaded with a numpy array containing (state, new_state, action, reward)
        self.experiences = deque(maxlen=2000)

    def reset(self):
        super().reset()

        self.last_estimated_move = None
        self.history = []
        self.prediction_history = []
        self.estimation_hits = 0.0
        self.estimation_misses = 0.0
        self.played_rounds = 0
        self.last_observations = None
        self.state = None

        self.experiences = deque(maxlen=2000)

        # Reset weights for the prediction model, so it can learn about a new agent
        if self.initial_prediction_weights is not None:
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

        new_state, new_preds = self.observation_to_state(self.last_observations)

        if self.state is not None:
            self.experiences.append(
                np.array([self.state, new_state, own_move.value, self.rewards[result]], dtype=object))

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

        self.prediction_model.fit(inputs, labels, verbose=0)

    def estimate_opponent_move(self) -> Move:
        """Runs the predictive model to estimate the opponents' next move"""
        obs = self.get_history_observations(self.move_batch_size)
        if obs is not None or (self.training and random.random() > self.epsilon):
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

            # Add the next prediction history

            inputs, pred_state = self.observation_to_state(observations)

            self.prediction_history.append(pred_state)
            self.state = inputs

            # Predict the next move
            next_move = self.gameplay_model.predict(inputs)
            next_move = self.moves[np.argmax(next_move)]
        else:
            next_move = random.choice(self.moves)
        return next_move

    @staticmethod
    def observation_to_state(observation):
        """Receives an observation and translates it into a state, for use with the network
        @:param observation Array-like [own_last_move, opponent_last_move, outcome, predicted_move, hit_ratio]
        @:return Tuple containing the Numpy array with the shape (1, 14) containing the observations for the gameplay,
        and for the predictions """
        own_last_move, opponent_last_move, outcome, predicted_move, hit_ratio = observation

        # Compute the categorical of the observation
        own_move = get_categorical_move(own_last_move)
        opposing_move = get_categorical_move(opponent_last_move)
        result = to_categorical(outcome.value, len(GameState))
        predicted_move = get_categorical_move(predicted_move)

        # Concatenate it all to a numpy array for feeding into the network
        state = np.concatenate([own_move, opposing_move, result, predicted_move, [hit_ratio]])
        state = state.reshape((1,) + state.shape)

        pred_state = np.concatenate([own_move, opposing_move, result])

        return state, pred_state

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

    def write_history(self, filepath):
        file = Path(filepath)
        file.touch()

        with open(file, 'a', newline='') as f:
            w = csv.writer(f, dialect='excel')
            # [own_move, opponent_move, result, self.last_estimated_move, self.hit_ratio()]
            w.writerow(["Own Move", "Opponent Move", "Result", "Last Estimated Move", "Hit Ratio"])
            w.writerows(self.history)

    def decay_epsilon(self, step):
        if self.epsilon_decay is not None:
            self.epsilon = self.epsilon_decay(step)
