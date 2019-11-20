import numpy as np

from keras.optimizers import SGD

from .. import encoders
from .. import goboard_fast as goboard
from .. import kerasutil
from ..agent import Agent

__all__ = [
    'GBAgent'
]


class GBAgent(Agent):
    def __init__(self, policy_model, value_model, encoder):
        Agent.__init__(self)
        self.policy_model = policy_model
        self.value_model = value_model
        self.encoder = encoder
        self.collector = None
        self.temperature = 1.0

        self.last_state_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        x = np.array([board_tensor])

        move_probs = self.policy_model.predict(x)[0]
        estimated_value = self.value_model.predict(x)[0][0]
        self.last_state_value = float(estimated_value)

        # Prevent move probs from getting stuck at 0 or 1.
        move_probs = np.power(move_probs, 1.0 / self.temperature)
        sum = np.sum(move_probs)
        move_probs = move_probs / np.sum(move_probs)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            true_move = goboard.Move.play(point)
            if not game_state.is_valid_move(true_move):
                true_move = goboard.Move.pass_turn()
            if self.collector is not None:
                self.collector.record_decision(
                    state=board_tensor,
                    action=point_idx,
                    estimated_value=estimated_value
                )
            return true_move
        # No legal, non-self-destructive moves less.
        return goboard.Move.pass_turn()

    def diagnostics(self):
        return {'value': self.last_state_value}


