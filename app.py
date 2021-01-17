from rps.agent.agents import RepeatAgent, HumanAgent
from rps.game.GameRunner import GameRunner
from rps_ml.baseline.rpscontest_agent import DLLU1Agent, MetaFixAgent
from rps_ml.ai_agent.ai_agent import AiAgent
from rps_ml.ai_agent.models import DefaultGameplayModel, DefaultPredictionModel

from rps_ml.ai_agent.training import RPSDojo
from pathlib import Path

import tensorflow as tf
import csv

NUM_ROUNDS = 1000
NUM_GAMES = 10
MODEL_LOCATION = 'SavedModels/BestGameplayModel'
DATA_LOCATION = ''

gameplay_model = DefaultGameplayModel()
prediction_model = DefaultPredictionModel()

gameplay_model.compile(loss='mse')
prediction_model.compile(optimizer=None, loss='categorical_crossentropy')


def run_training():
    trainee = AiAgent(prediction_model, gameplay_model, epsilon=0.2)
    trainer = MetaFixAgent()

    dojo = RPSDojo(trainee=trainee,
                   trainer=trainer,
                   episodes=NUM_GAMES,
                   optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.MeanSquaredError(),
                   discount=0.95,
                   rounds_in_episode=NUM_ROUNDS,
                   trainee_logging_path=DATA_LOCATION + 'action_history.csv'
                   )

    dojo.run_training()

    gameplay_model.save(MODEL_LOCATION, include_optimizer=False)

    dojo.write_history(DATA_LOCATION + 'training_history.csv')


def test_simple_match():
    pai = AiAgent(prediction_model=prediction_model,
                  gameplay_model=gameplay_model)

    p1 = pai
    p2 = MetaFixAgent()

    game = GameRunner(p1, p2, num_rounds=NUM_ROUNDS)

    score = {p1: 0, p2: 0}
    for i in range(NUM_GAMES):
        print("\rPlaying game %d / %d" % (i + 1, NUM_GAMES), end="")
        while not game.game_over:
            game.play_round()
        winner = game.winner
        if winner is not None:
            score[winner] += 1
        game.reset()
    print()
    print(score)


run_training()
