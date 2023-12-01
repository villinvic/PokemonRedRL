import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyboy.pyboy import PyBoy

from pkmn_env.save_state_info import PokemonStateInfo
from python_utils.collections import DefaultOrderedDict
from pkmn_env.enums import *


# Module based on go-explore algorithm

# Simplified for our setting

class GoExplorePokemon:

    TIMES_SEEN = "times_seen"
    TIMES_CHOSEN_SINCE_NEW_WEIGHT = "times_chosen_since_new_weight"
    TIMES_CHOSEN = "times_chosen"

    def __init__(self, environment, path: Path, relevant_state_features,
                 sample_base_state_chance=0.2, recompute_score_freq=1, rendering=False):

        self.environment = environment
        self.console = environment.pyboy

        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.base_state_info = environment.base_state_info
        self.states = DefaultOrderedDict(PokemonStateInfo)

        self.reset_count = 0

        self.stat_weights = {
            GoExplorePokemon.TIMES_CHOSEN_SINCE_NEW_WEIGHT: 0.,
            GoExplorePokemon.TIMES_CHOSEN: 1.,
            GoExplorePokemon.TIMES_SEEN: 20.,
        }

        self.state_stats = defaultdict(lambda : {
            GoExplorePokemon.TIMES_CHOSEN_SINCE_NEW_WEIGHT: 0.,
            GoExplorePokemon.TIMES_CHOSEN                 : 0.,
            GoExplorePokemon.TIMES_SEEN                   : 0,
        })

        self.sample_base_state_chance = sample_base_state_chance
        self.relevant_state_features = relevant_state_features
        self.recompute_score_freq = recompute_score_freq
        self.rendering = rendering

    def add_starting_point(self, game_stats):
        # We can add more params to the state
        # for now, we stick to badges and map_id

        features = tuple(game_stats[feature][-1] for feature in self.relevant_state_features)
        hash_name = hash(features)
        file_base = f"{hash_name}"
        state_name = file_base + ".state"
        state_info_name = file_base + "_info.pkl"

        _, _, files = next(os.walk(self.path))

        if np.random.random() < 1e-3 or state_name not in files:

            save_state_path = self.path / state_name
            info_path = self.path / state_info_name
            info = PokemonStateInfo(save_path=self.path / file_base,
                                    highest_opponent_level_so_far=self.environment.highest_opponent_level_so_far,
                                    visited_maps=self.environment.visited_maps
                                    )
            with open(save_state_path, "wb") as f:
                self.console.save_state(f)
            with open(info_path, "wb") as f:
                pickle.dump(info.to_dict(), f)

    def read_session_states(self):
        _, _, files = next(os.walk(self.path))

        for file in files:
            if file.endswith(".state"):
                save_name = file[:-6]
                if save_name not in self.states:
                    self.states[save_name].save_path = self.path / save_name

    def update_stats(self, game_stats):

        # has to be hashable
        state_indexer = hash(tuple(game_stats[feature][-1] for feature in self.relevant_state_features))
        # update count
        self.state_stats[state_indexer][GoExplorePokemon.TIMES_SEEN] += 1

    def compute_scores(self):
        scores = []
        for state_hash, info in self.states.items():

            state_stats = self.state_stats[int(state_hash)]

            score = 0

            for stat_name, value in state_stats.items():
                score += self.stat_weights[stat_name] * np.sqrt(1 / (value + 1e-3))

            scores.append(score + 3e-5)

        return scores

    def sample_starting_point(self, scores):
        if np.random.random() < self.sample_base_state_chance:
            return self.base_state_info

        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        print(probs)

        starting_point_name = np.random.choice(list(self.states.keys()), p=probs)

        self.state_stats[starting_point_name][GoExplorePokemon.TIMES_CHOSEN] += 1

        return self.states[starting_point_name]

    def __call__(self):

        if self.rendering:
            self.reset_console(self.base_state_info)
        else:
            scores = self.compute_scores()
            self.reset_console(self.sample_starting_point(scores))

    def reset_console(self, state: PokemonStateInfo):
        # restart game, skipping credits
        with open(state.get_save_path(), "rb") as f:
            self.console.load_state(f)

        if state.highest_opponent_level_so_far is not None:
            # Base state
            # -> no pkl saved for base state
            state.send_to_env(self.environment)
        else:
            with open(state.get_info_path(), "rb") as f:
                PokemonStateInfo.dict_to_env(self.environment, pickle.load(f))






