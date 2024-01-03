import argparse
import io
import pickle
import sys
import time
import uuid
import os
from copy import deepcopy
from functools import partial
from typing import List, Callable

import cv2
from collections import defaultdict
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib
import mediapy as media
import pandas as pd

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from pkmn_env.go_explore import GoExplorePokemon
from pkmn_env.save_state_info import PokemonStateInfo
#from python_utils.collections import DefaultOrderedDict

from pkmn_env.enums import *


class PkmnRedEnvNoRender(Env):


    # Observation names

    LOGGABLE_VALUES = (
        MAPS_VISITED,
        BADGE_SUM,
        MONEY,
        TOTAL_LEVELS,
        TOTAL_EXPERIENCE,
        SEEN_POKEMONS,
        CAUGHT_POKEMONS,
        TOTAL_BLACKOUT,
        TOTAL_EVENTS_TRIGGERED,
        NUM_BALLS_USED,
        NUM_HEALING_ITEMS_USED,
        NUM_BALLS_BOUGHT,
        NUM_HEALING_ITEMS_BOUGHT,
    )

    def __init__(
            self,
            config=None
    ):

        self.worker_index = 1 if not hasattr(config, "worker_index") else config.worker_index
        self.init_state = config['init_state']
        self.s_path = config['session_path']
        self.s_path.mkdir(exist_ok=True)

        self.pokemon_centers = {
            41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 174, 182,
        }
        self.poke_marts = {
            42, 56, 67, 91, 150, 152, 172, 173, 180
        }

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
            #WindowEvent.PASS
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]


        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.observation_space = spaces.Discrete(1)

        self.reward_function_config = {
            TOTAL_EXPERIENCE         :   1.,  # 0.5
        }

        self.pyboy = PyBoy(
            config['gb_path'],
            disable_input=True,
            disable_renderer=not config["render"],
            hide_window=not config["render"],
            window_type="SDL2" if config["render"] else "headless",
        )

        #self.pyboy.set_emulation_speed(0)

        self.game_stats = None
        self.step_count = 0
        self.episode_reward = 0
        self.visited_maps = {40}

        self.base_starting_point = open(self.init_state, "rb")

        self.act_freq = 18

        # self.go_explore = GoExplorePokemon(
        #     environment=self,
        #     path=self.s_path / "go_explore",
        #     relevant_state_features=(BADGE_SUM, MAP_ID), # EVENTS ?
        #     sample_base_state_chance=1.0,
        #     recompute_score_freq=1,
        #     rendering=False
        # )
        #self.init_knn()

        self.inited = 0

    def tick(self):
        self.pyboy.tick()


    def run_action_on_emulator(self, action):

        # press button then release after some steps
        console_input = self.valid_actions[action]
        self.pyboy.send_input(console_input)

        #act_freq = self.act_freq if (self.step_count < 2 or not self.game_stats[IN_BATTLE][-1]) else self.act_freq * 4

        for i in range(self.act_freq):
            # release action, so they are stateless

            if i == 8:

                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])

                if 3 < action < 6:
                    # release button
                    self.pyboy.send_input(self.release_button[action - 4])

                if console_input == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

            self.tick()

    def reset(self, options=None, seed=None):

        del self.game_stats


        if options is None:
            self.base_starting_point.seek(0)
            self.pyboy.load_state(self.base_starting_point)

            self.game_stats = defaultdict(list)

            self.step_count = 0
            self.episode_reward = 0
            self.visited_maps = set()
        else:

            start_point = options.get("state", self.base_starting_point)
            self.base_starting_point.seek(0)
            self.pyboy.load_state(start_point)
            self.game_stats = options.get("game_stats", defaultdict(list))
            if MAP_ID in self.game_stats:
                print(self.game_stats[MAP_ID][-1], self.read_map_id())
            self.episode_reward = options.get("episode_reward", 0.)
            self.visited_maps = options.get("visited_maps", set())
            self.step_count = options.get("step_count", 0)


        return 0, {}

    def update_observed_stats(self):

        self.game_stats[MONEY].append(self.read_money())
        party_levels = self.read_party_levels()

        in_battle = self.read_in_battle()
        self.game_stats[IN_BATTLE].append(in_battle)
        self.game_stats[PARTY_LEVELS].append(party_levels)

        opp_level = self.read_opponent_level()
        # in_battle = opp_level not in (0, 255)

        self.game_stats[TOTAL_LEVELS].append(sum(party_levels))
        party_experience = self.read_party_experience()
        self.game_stats[PARTY_EXPERIENCE].append(party_experience)
        self.game_stats[TOTAL_EXPERIENCE].append(sum(party_experience))

        num_balls, num_healing_items = self.read_inventory()
        if self.step_count == 0:
            dballs = 0
            dhealing_items = 0
        else:

            dballs = num_balls - self.game_stats[NUM_BALLS][-1]
            dhealing_items = num_healing_items - self.game_stats[NUM_HEALING_ITEMS][-1]

        prev_count = 0 if self.step_count == 0 else self.game_stats[NUM_BALLS_USED][-1]
        self.game_stats[NUM_BALLS_USED].append(prev_count + np.maximum(0, -dballs))
        prev_count = 0 if self.step_count == 0 else self.game_stats[NUM_BALLS_BOUGHT][-1]
        self.game_stats[NUM_BALLS_BOUGHT].append(prev_count + np.maximum(0, dballs))

        prev_count = 0 if self.step_count == 0 else self.game_stats[NUM_HEALING_ITEMS_USED][-1]
        self.game_stats[NUM_HEALING_ITEMS_USED].append(prev_count + np.maximum(0, -dhealing_items))
        prev_count = 0 if self.step_count == 0 else self.game_stats[NUM_HEALING_ITEMS_BOUGHT][-1]
        self.game_stats[NUM_HEALING_ITEMS_BOUGHT].append(prev_count + np.maximum(0, dhealing_items))

        self.game_stats[NUM_BALLS].append(num_balls)
        self.game_stats[NUM_HEALING_ITEMS].append(num_healing_items)

        badges = self.read_badges()
        self.game_stats[BADGES].append(badges)
        self.game_stats[BADGE_SUM].append(sum(badges))
        self.game_stats[SEEN_POKEMONS].append(self.read_seen())
        self.game_stats[CAUGHT_POKEMONS].append(self.read_caught())
        total_events = sum(self.read_events())
        self.game_stats[TOTAL_EVENTS_TRIGGERED].append(total_events)
        self.game_stats[PARTY_FILLS].append(self.read_party_fills())
        self.game_stats[SENT_OUT].append(self.read_sent_out())
        party_health = self.read_party_health()
        self.game_stats[PARTY_HEALTH].append(party_health)

        map_id = self.read_map_id()
        self.game_stats[MAP_ID].append(map_id)

        self.game_stats[BLACKOUT].append(
            self.step_count > 2
            and
            (self.game_stats[IN_BATTLE][-2] and not in_battle)
            and
            (self.game_stats[MAP_ID][-3] != map_id)

        )

        tmp = self.visited_maps | {map_id}
        self.game_stats[MAPS_VISITED].append(len(tmp))

        pos = self.read_pos()
        curr_coords = pos + [map_id]
        self.game_stats[COORDINATES].append(curr_coords)

        
    def step(self, action):

        self.run_action_on_emulator(action)
        self.update_observed_stats()
        self.step_count += 1

        reward = self.get_game_state_reward()
        self.episode_reward += reward

        return 0, reward, False, False, {}

    def get_game_state_reward(self):
        """
        proposed reward function:
            - knn (reset every important event, hopefully)
            - sum of cqrt(exp) over pokemons
            - badges

        Exploration should be motivated through knn (navigation)
        and going to areas with higher level pokemons (more exp)
        :return: computed rewards sum
        """

        rewards = {}

        if self.step_count >= 2:
            curr_coords = tuple(self.game_stats[COORDINATES][-1])

            # we gain more experience as game moves on:
            total_delta_exp = 0
            #total_healing = 0
            highest_party_level = max(self.game_stats[PARTY_LEVELS][-1])

            if curr_coords not in self.pokemon_centers and self.game_stats[IN_BATTLE][-1]:

                for i in range(6):

                    total_delta_exp += np.maximum(
                        (self.game_stats[PARTY_EXPERIENCE][-1][i]
                        - self.game_stats[PARTY_EXPERIENCE][-2][i]) * int(self.game_stats[PARTY_EXPERIENCE][-2][i] != 0.)
                        , 0.
                    ) / highest_party_level**3

            # if not any(self.game_stats[BLACKOUT][-2:]) and curr_coords[-1] in self.pokemon_centers:
            #     for i in range(6):
            #         # We need to make sure one of the pokemons wasnt KO and healed
            #         total_healing += int(
            #             self.game_stats[PARTY_HEALTH][-1][i]-self.game_stats[PARTY_HEALTH][-2][i]
            #             > 0.05
            #             and
            #             self.game_stats[PARTY_HEALTH][-2][i] > 0
            #         )

            rewards.update(**{
                TOTAL_EXPERIENCE: total_delta_exp,
                # BLACKOUT: self.game_stats[BLACKOUT][-1],
                # BADGE_SUM: (
                #     np.maximum(self.game_stats[BADGE_SUM][-1] - self.game_stats[BADGE_SUM][-2], 0.) * (self.game_stats[BADGE_SUM][-2] + 1)
                # ),
                # SEEN_POKEMONS : (
                #     np.maximum(self.game_stats[SEEN_POKEMONS][-1] - self.game_stats[SEEN_POKEMONS][-2],
                #                0.) * (self.game_stats[SEEN_POKEMONS][-2])
                # ),
                # TOTAL_EVENTS_TRIGGERED: (
                #         np.maximum(self.game_stats[TOTAL_EVENTS_TRIGGERED][-1]
                #         - self.game_stats[TOTAL_EVENTS_TRIGGERED][-2], 0) * (self.game_stats[TOTAL_EVENTS_TRIGGERED][-2] - 10)
                # ),
                #
                # MAPS_VISITED: (
                # (self.game_stats[MAPS_VISITED][-1] - self.game_stats[MAPS_VISITED][-2]) * self.game_stats[MAPS_VISITED][-2]
                # ),
                #
                #
                # PARTY_HEALTH: int(total_healing > 0),
                #
                # MONEY : int(self.game_stats[MONEY][-1] - self.game_stats[MONEY][-2] > 50
                #             and
                #             curr_coords[-1] not in self.poke_marts
                #             ),

            })


        total_reward = 0
        for reward_name, reward in rewards.items():
            scaled_reward = reward * self.reward_function_config[reward_name]
            self.game_stats["reward_"+reward_name].append(scaled_reward)
            total_reward += scaled_reward

        return total_reward


    def get_stats(self) -> dict:

        stats = {"episode_reward": self.episode_reward}
        stats.update(**{
            k: metric[-1] for k, metric in self.game_stats.items()
        })

        return stats

    def game_state(self):

        file_like_object = io.BytesIO()
        file_like_object.seek(0)
        self.pyboy.save_state(file_like_object)

        state = {
            "state": file_like_object,
            "visited_maps": self.visited_maps.copy(),
            "episode_reward": self.episode_reward,
            "game_stats": deepcopy(self.game_stats),
            "step_count": self.step_count
        }
        return state

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_double(self, start_addr):
        return 256 * self.read_m(start_addr) + self.read_m(start_addr + 1)

    def read_triple(self, start_addr):
        return 256 * 256 * self.read_m(start_addr) + 256 * self.read_m(start_addr + 1) + self.read_m(start_addr + 2)

    def read_bcd(self, addr):
        num = self.read_m(addr)
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)

    def read_money(self):
        return (100 * 100 * self.read_bcd(0xD347) +
                100 * self.read_bcd(0xD348) +
                self.read_bcd(0xD349))

    def read_map_id(self):
        return self.read_m(0xD35E)

    def read_party_ptypes(self) -> List:
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def read_party_fills(self) -> List:
        """

        :return: 1 if pokemon on that slot, else 0
        """

        return [
            int(self.read_m(addr) not in (0, 255))
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]

    def read_party_health(self) -> List:
        """
        :return: current party health fractions
        """

        return [
            self.read_double(curr_hp_addr)/(self.read_double(max_hp_addr)+1e-8)
            for curr_hp_addr, max_hp_addr in
            [(0xD16C, 0xD18D), (0xD198, 0xD1B9), (0xD1C4, 0xD1E5), (0xD1F0, 0xD211), (0xD21C, 0xD23D), (0xD248, 0xD269)]
        ]

    def read_party_experience(self) -> List:
        return [
            self.read_triple(addr)
            for addr in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
        ]

    def read_party_levels(self) -> List:

        return [
            self.read_m(addr)
            for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]

    def read_party_evs(self):
        pass

    def read_caught(self) -> int:

        total_caught = sum([
            self.read_m(addr).bit_count() for addr in range(0xD2F7, 0xD309)
        ])
        return total_caught

    def read_seen(self) -> int:

        total_seen = sum([
            self.read_m(addr).bit_count() for addr in range(0xD30A, 0xD31C)
        ])
        return total_seen

    def read_badges(self) -> List:

        # badges are binary switches here:
        binary = self.read_m(0xD356).bit_count()

        return [int(i < binary) for i in range(8)]

    def read_events(self) -> List:
        return [self.read_m(i).bit_count() for i in range(0xD747, 0xD886)]

    def read_extensive_events(self) -> List:
        flags_indices = []

        for i in range(0xD747, 0xD886):
            value = self.read_m(i)

            index = 0
            while value:
                if value & 1:
                    flags_indices.append((i - 0xD747) * 8 + index)
                value >>= 1
                index += 1

        return flags_indices

    def read_pos(self) -> List:
        return [self.read_m(0xD362), self.read_m(0xD361)]

    def read_walk_animation(self) -> int:
        return self.read_m(0xC108)

    def read_orientation(self) -> int:
        return self.read_m(0xC109)

    def read_opponent_level(self) -> int:
        return self.read_m(0xCFF3)

    def read_sent_out(self) -> List:
        return [int(i == self.read_m(0xCC2F)) for i in range(6)]

    def read_in_battle(self) -> int:
        return int(self.read_m(0xD057) != 0)

    def read_inventory(self) -> List:

        pokeballs = 0
        healing_items = 0
        for addr in range(0xD31E, 0xD346, 2):
            item_id = self.read_m(addr)
            count = self.read_m(addr + 1)

            if 0 < item_id < 5:
                pokeballs += count
            elif 15 < item_id < 21:
                healing_items += count

        return [pokeballs, healing_items]

    def read_textbox_id(self) -> int:
        return self.read_m(0xD125)

    def read_party_menu(self) -> int:
        # Actually reads the sprite animation ids
        return self.read_m(0xD09B)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Action sequence renderer',
        description='Renders an action sequence')

    parser.add_argument('sequence_path')
    parser.add_argument('-s', '--speed',  default=1, type=float)

    args = parser.parse_args()

    conf = dict(
        render=True,
        debug=False,
        session_path=Path("red_tests"),
        headless=False,
        init_state="deepred_post_parcel_pokeballs",
        max_steps=1024,
        fast_video=False,
        save_video=False,
        additional_steps_per_episode=1,
        save_final_state=True,
        gb_path="pokered.gbc",

    )

    env = PkmnRedEnvNoRender(conf)
    env.pyboy.set_emulation_speed(args.speed)

    obs, _ = env.reset()
    done = False

    times = []
    with open(args.sequence_path, "rb") as f:
        actions = pickle.load(f)

    t = time.time()
    for action in actions:
        if action < 7:
            env.step(action)
            t2 = time.time()
            times.append(t2-t)
            print(t2-t)
            t = t2
    print(np.max(times), np.mean(times), np.min(times))











