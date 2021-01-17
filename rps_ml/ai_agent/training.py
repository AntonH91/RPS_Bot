import tensorflow as tf

from rps.agent.base_agent import RPSAgent
from rps_ml.ai_agent.ai_agent import AiAgent
from rps.constants import GameState, Move

from rps.game.GameRunner import GameRunner
import rps.utils.utils as utils
import numpy as np

from pathlib import Path
import csv


class RPSDojo():

    def __init__(self, trainee: AiAgent, trainer: RPSAgent, episodes: int,
                 optimizer: tf.keras.optimizers.Optimizer, loss: tf.keras.losses.Loss,
                 batch_size=64, discount=0.9, rounds_in_episode=1000,
                 trainee_logging_path=None) -> None:
        """Instantiates a new training setup for the ML Agent

        @:param trainee The AI Agent to be trained
        @:param trainer The RPSAgent which the AI Agent should train against
        @:param episodes The number of episodes to run
        @:param optimizer A Keras optimizer for use with training
        @:param loss A Keras Loss funmodection for use with training
        @:param batch_size The batch size to use for training the model
        @:param discount The discount to apply for older actions
        @:param rewards A Set containing the various GameStates to compute a reward for the agent."""

        assert isinstance(episodes, int), "Episodes must be a given number."
        assert episodes > 0, "Episodes must be greater than 0"

        self.trainee = trainee
        self.trainer = trainer

        self.episodes = episodes
        self.optimizer = optimizer
        self.loss = loss
        self.discount = discount
        self.batch_size = batch_size
        self.game = GameRunner(player_one=trainee,
                               player_two=trainer,
                               num_rounds=rounds_in_episode)

        self.best_reward = 0
        self.best_weights = None

        self.training_history = []
        self.trainee_logging_path = trainee_logging_path

    def run_training(self):
        model = self.trainee.gameplay_model
        loss = -1
        for i in range(self.episodes):

            game_history = []
            total_reward = 0
            round_count = 0

            while not self.game.game_over:
                round_count += 1
                trainee_move, trainer_move, winner, game_over = self.game.play_round()

                trainee_state = utils.check_winner(trainee_move, trainer_move)
                reward = self.trainee.rewards[trainee_state]

                game_history.append((trainee_move, trainer_move, reward))
                total_reward += reward

                if round_count % 20 == 0:
                    print("\rTraining game: %d / %d (%d pct) - Total reward: %d" %
                          (i + 1, self.episodes, (round_count / self.game.num_rounds) * 100, total_reward),
                          end="")

            loss = self.training_step()

            self.training_history.append((float(loss), total_reward))

            # total_reward = tf.reduce_sum(self.trainee.experiences, axis=4)

            if self.best_weights is None or total_reward > self.best_reward:
                self.best_reward = total_reward
                self.best_weights = model.get_weights()

            print("\rTraining game: %d / %d (%d pct) - Total reward: %d , Loss: %d" %
                  (i + 1, self.episodes, (round_count / self.game.num_rounds) * 100, total_reward, loss),
                  end="")

            if self.trainee_logging_path is not None:
                self.trainee.write_history(self.trainee_logging_path)

            self.game.reset()
            print()

        model.set_weights(self.best_weights)

    # Gathered from
    # https://colab.research.google.com/github/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb#scrollTo=K912ogLfbdYX
    def training_step(self, batch_size=32):
        model = self.trainee.gameplay_model
        experiences = self.sample_experiences()

        states, new_states, actions, rewards = experiences

        next_q = model.predict(new_states)
        max_next_values = np.max(next_q, axis=2)
        rewards = rewards.reshape(rewards.shape + (1,))
        target_q = (rewards + self.discount * max_next_values)
        # target_q = target_q.reshape(-1, 1)

        # Offset actions by - 1 since the value of it is based on the Move enum
        mask = tf.one_hot(actions - 1, len(Move))

        with tf.GradientTape() as tape:
            all_q = model(states)
            q_values = tf.reduce_sum(all_q * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss(target_q, q_values))

        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    def sample_experiences(self, batch_size=32):
        experiences = self.trainee.experiences

        indices = np.random.randint(len(experiences), size=batch_size)
        batch = [experiences[index] for index in indices]
        states, new_states, actions, rewards = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(4)]
        return states, new_states, actions, rewards

    def write_history(self, filepath):

        file = Path(filepath)
        file.touch(exist_ok=True)

        with open(filepath, 'a', newline='') as f:
            w = csv.writer(f, dialect='excel')
            w.writerow(["Loss", "Reward"])
            w.writerows(self.training_history)
