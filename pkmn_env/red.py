import sys
import time
import uuid
import os
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
from python_utils.collections import DefaultOrderedDict

from pkmn_env.enums import *


# TODO:
#       - scale up novelty for each map we visit
#       - run with stable lstm


class VariableGetter:

    def __init__(
            self,
            *,
            name: str,
            dim: int = 1,
            scale: float = 1.,
            post_process_fn: Callable = lambda value: value
    ):

        self.dim = dim
        self.scale = scale
        self.name = name
        self.post_process_fn = post_process_fn

    def grab(
            self,
            observation_history: DefaultOrderedDict
    ) -> any:

        return observation_history[self.name][-1]


    def __call__(
            self,
            *,
            observation_history: DefaultOrderedDict,
            post_processed_array: np.ndarray,
            index: int
    ) -> int:
        """

        :param observation_history: observed history so far
        :param post_processed: array to insert value in
        :param index: starting index of our value
        :return: next index
        """
        v = self.grab(observation_history)

        if self.dim == 1:
            post_processed_array[index] = self.scale * self.post_process_fn(v)
        else:
            post_processed_array[index: index + self.dim] = v
            post_processed_array[index: index + self.dim] = (
                    self.scale * self.post_process_fn(post_processed_array[index: index + self.dim])
            )


        return index + self.dim


class PkmnRedEnv(Env):


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
        self.debug = config['debug']
        self.s_path = config['session_path']
        self.headless = config['headless']
        self.num_elements = config['knn_elements']
        self.init_state = config['init_state']
        self.max_steps = config['max_steps']
        self.save_video = config['save_video'] and self.worker_index == 1
        self.fast_video = config['fast_video'] and self.worker_index == 1
        self.additional_steps_per_episode = config['additional_steps_per_episode']

        self.save_final_state = config['save_final_state'] and self.worker_index == 1

        self.screen_shape = (36, 40)  # (48, 56) # (72, 80)
        self.stacked_frames = 3
        self.screen_observation = np.zeros((self.screen_shape[0]*self.stacked_frames, self.screen_shape[1], 1), dtype=np.uint8)
        self.similar_frame_dist = config['sim_frame_dist']
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)

        self.metadata = {"render.modes": []}
        self.reward_range = (-np.inf, np.inf)

        self.pokemon_centers = {
            41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 174, 182,
        }
        self.poke_marts = {
            42, 56, 67, 91, 150, 152, 172, 173, 180
        }

        # self.ball_price_to_item_value = {
        #     200: 1, # pokeball
        #     600: 2, # greatball
        #     1200: 1.5, # ultraball
        #     0: 2, # ball was obtained
        # }
        #
        # self.heal_price_to_item_value = {
        #     3000 : 4,  # full restore
        #     2500: 4,  # max potion
        #     1500 : 3,  # hyper potion
        #     1200: 1.5,  # ultraball
        #     0   : 5,  # ball was obtained
        # }

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #WindowEvent.PRESS_BUTTON_START,
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

        self.observed_stats_config = [
            VariableGetter(
                dim=8,
                name=BADGES
            ),
            VariableGetter(
                name=MONEY,
                scale=1e-4
            ),
            VariableGetter(
                name=SEEN_POKEMONS,
                scale=1e-2
            ),
            VariableGetter(
                name=CAUGHT_POKEMONS,
                scale=0.1
            ),
            VariableGetter(
                dim=6,
                name=PARTY_HEALTH
            ),
            VariableGetter(
                name=IN_BATTLE
            ),
            VariableGetter(
                name=NUM_BALLS,
                scale= 0.2,
                post_process_fn=lambda x: np.clip(x, 0., 2.)
            ),
            VariableGetter(
                name=NUM_HEALING_ITEMS,
                scale=0.2,
                post_process_fn=lambda x: np.clip(x, 0., 2.)
            ),
            # Needs 1_200_000 exp max to reach level 100
            # We will need better information that that
            # VariableGetter(
            #     dim=6,
            #     name=PkmnRedEnv.PARTY_EXPERIENCE,
            #     post_process_fn=np.cbrt,
            #     scale=1/np.cbrt(1e6)
            # ),
            # VariableGetter(
            #     dim=2,
            #     name=PkmnRedEnv.ENTRANCE_DELTA_POS,
            #     scale=0.05,
            # ),
            VariableGetter(
                dim=6,
                name=PARTY_LEVELS,
                scale=0.02,
            ),
            VariableGetter(
                name=LEVEL_FRAQ,
                scale=0.5,
                post_process_fn=lambda x: np.maximum(x, 2.)
            ),
            VariableGetter(
                name=TOTAL_EVENTS_TRIGGERED,
                scale=0.02 #319
            ),
            VariableGetter(
                name=MAPS_VISITED,
                scale=0.05
            ),
            VariableGetter(
                dim=6,
                name=PARTY_FILLS,
            ),
            VariableGetter(
                dim=6,
                name=SENT_OUT
            )
            # VariableGetter(
            #     dim=5,
            #     name=PkmnRedEnv.SURROUNDING_TILES_VISITATION,
            #     scale=0.2
            # ),
            #VariableGetter(
            #     name=PkmnRedEnv.NOVELTY_COUNT,
            #     scale=0.01
            # ),
            # VariableGetter(
            #     dim=8,
            #     name="party_fills",
            #     post_process_fn=np.cbrt,
            #     scale=1 / np.cbrt(1e6)
            # ),

            # TODO :
            # Group items (healing, pokeballs) and count
            # Give reward for money ?
            # observe opponent level
            # Keep track of the highest leveled pokemon encountered and instead of observing level, observe fraction
            # between our level and highest opponent level.
        ]

        self.reward_function_config = {
            BLACKOUT                 :   - 0.15,
            SEEN_POKEMONS            :   0.1,
            TOTAL_EXPERIENCE         :   3.,  # 0.5
            BADGE_SUM                :   100.,
            MAPS_VISITED             :   0.05, # 3.
            TOTAL_EVENTS_TRIGGERED   :   0.03, # TODO : bugged
            MONEY                    :   8.,
            #COORDINATES              :   - 5e-4,
            # COORDINATES + "_NEG"     :   0.003 * 0.9,
            # COORDINATES + "_POS"     :   0.003,
            PARTY_HEALTH             :   3.,

            #GOAL_TASK                :  0.5,

            #ITEMS                    :  0.1,


        }

        self.additional_features_shape = (
            sum(v.dim for v in self.observed_stats_config),
        )
        self.observed_stats = np.zeros(self.additional_features_shape, dtype=np.float32)

        #self.triggered_event_flags = np.zeros((0xD886 - 0xD747) * 8, dtype=np.uint8)

        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(self.screen_shape[0]*self.stacked_frames, self.screen_shape[1], 1,), dtype=np.uint8),
            "stats": spaces.Box(low=-np.inf, high=np.inf, shape=self.additional_features_shape, dtype=np.float32),
            #"flags": spaces.Box(low=0, high=1, shape=(len(self.triggered_event_flags),), dtype=np.uint8),
            "coordinates": spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8),
            "moved": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "allowed_actions": spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.uint8),

        })

        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=True,
            window_type='headless' if config['headless'] else 'SDL2',
            hide_window='--quiet' in sys.argv,
        )

        self.screen = self.pyboy.botsupport_manager().screen()
        self.pyboy.set_emulation_speed(0 if config['headless'] else 6)

        self.knn_index = None
        self.game_stats: DefaultOrderedDict = None
        self.distinct_frames_observed = 0
        self.maximum_experience_in_party_so_far = 0
        self.step_count = 0
        self.max_steps_noised = 0
        self.episode_reward = 0
        self.visited_coordinates = defaultdict(lambda: 0)
        self.entrance_coords = (5, 3, 40)
        self.last_reward_dict = {}
        self.last_walked_coordinates = []
        self.full_frame_writer = None

        self.goal_task_timeout_steps = 256
        self.current_goal = None
        self.task_timesteps = 0
        self.stuck_count = 0
        self.target_symbol_mask = np.zeros((8, 8, 1), dtype=np.uint8)
        self.target_symbol_mask[1 : -1, 1 : -1] = 1
        self.target_symbol_mask_debug = np.zeros((16, 16, 3), dtype=np.uint8)
        self.target_symbol_mask_debug[2 : -2, 2 : -2] = 1

        self.base_state_info = PokemonStateInfo(
            save_path=Path(self.init_state),
            latest_opp_level=3,
            visited_maps={37, 38, 39, 40}  # red (first and second floor) and blue houses
        )

        if config["headless"]:
            # we are not rendering the game live
            self.act_freq = 19

        else:

            self.act_freq = 19

        self.go_explore = GoExplorePokemon(
            environment=self,
            path=self.s_path / "go_explore",
            relevant_state_features=(BADGE_SUM, MAP_ID), # EVENTS ?
            sample_base_state_chance=1.0,
            recompute_score_freq=1,
            rendering=not config["headless"] # tests
        )

        #self.init_knn()

        self.inited = 0

    def _get_obs(self):

        obs = {
            "stats"  :   self.get_observed_stats(),
            "coordinates": self.get_coordinates(),
            "moved"      : self.get_moved(),
            "allowed_actions": self.get_allowed_actions(),
        }

        # if self.current_goal is None or (self.goal_task_timeout_steps - self.task_timesteps <= 0):
        #
        #     x, y, map_id = tuple(self.game_stats[COORDINATES][-1])
        #
        #     if map_id in {0, 37, 38, 39, 40}:
        #         self.current_goal = (0, 0, -1)
        #         self.task_timesteps = self.goal_task_timeout_steps
        #     else:
        #         df = np.random.randint(3, 8) * np.random.choice([-1, 1])
        #         dc = np.random.randint(0, 3) * np.random.choice([-1, 1])
        #         dd = [df, dc]
        #         np.random.shuffle(dd)
        #
        #         self.current_goal = (x + dd[0], y + dd[1], map_id)
        #         self.task_timesteps = 0
        #
        # if not self.game_stats[IN_BATTLE][-1]:
        #     self.task_timesteps += 1

        obs["screen"] = self.render()
        return obs

    def run_action_on_emulator(self, action):

        # press button then release after some steps
        console_input = self.valid_actions[action]
        self.pyboy.send_input(console_input)
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
            self.pyboy.tick()

            if self.save_video and not self.fast_video:
                self.add_video_frame()

        if self.skippable_battle_frame():
            # Without speedup rom and skipping the frames, agent has to take 14-20 meaningless actions per turn.
            self.skip_battle_frames()

        if self.save_video and self.fast_video:
            self.add_video_frame()



    def skippable_battle_frame(self):
        # We skip the frame if we are in battle, have a certain box id open or if we are in the party menu.
        return (
                self.read_in_battle()
                and
                not (
                self.read_textbox_id() in {11, 12, 13, 20}
                or
                self.read_party_menu() > 0
                )
        )

    def skip_battle_frames(self):
        c = 0
        for i in range(18 * 32):
            # Skip battle animations
            self.pyboy.tick()
            if not self.skippable_battle_frame():
                # Some message box have id 11 midturn, but they are automatically scrolled.
                # So we just wait them out
                c += 1
                if c == 5:
                    # Needs one more action when battle ends
                    break


    def reset(self, options=None, seed=None):

        del self.game_stats

        self.game_stats = DefaultOrderedDict(list)
        #self.triggered_event_flags = np.zeros((0xD886 - 0xD747) * 8, dtype=np.uint8)
        self.last_reward_dict = {}

        #  We init only once now
        # self.init_knn()
        # self.distinct_frames_observed = 0

        self.step_count = 0
        self.maximum_experience_in_party_so_far = 0
        self.episode_reward = 0
        self.screen_observation[:] = 0
        self.visited_maps = {37, 38, 39}
        self.entrance_coords = (5, 30, 40)
        self.latest_opp_level = 5
        self.visited_coordinates = defaultdict(lambda: 0)
        self.last_walked_coordinates = []

        self.current_goal = None
        self.task_timesteps = 0

        # we restart the game at a random, relevant state
        self.go_explore.read_session_states()
        self.go_explore()

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'video_{self.reset_count}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()

        noise = int(0.1 * self.max_steps)
        if self.inited > 2:
            self.max_steps_noised = self.max_steps# + (
            #(int(0.2 * (self.worker_index / 124) * self.max_steps) // 2048) * 2048
            #) # np.random.randint(-noise, noise)
        else:
            self.inited += 1

            self.max_steps_noised = 2048 + (2048 * self.worker_index % self.max_steps)

        return self._get_obs(), {}

    def preprocess_screen(self, screen):
        # don't care about order, image is already in gray
        grayscale_screen = np.uint8(
                         0.299 * screen[:, :, 0]
                         + 0.587 * screen[:, :, 1]
                         + 0.114 * screen[:, :, 2]
                 )

        grayscale_downsampled_screen = cv2.resize(
            grayscale_screen,
            tuple(reversed(self.screen_shape)),
            interpolation=cv2.INTER_AREA,
        )[:, :, np.newaxis]

        # Render target
        # x, y, goal_map_id = self.current_goal
        # curr_x, curr_y, curr_map_id = self.game_stats[COORDINATES][-1]
        #
        # if goal_map_id == curr_map_id and not self.game_stats[IN_BATTLE][-1]:
        #     dx = (x - curr_x)
        #     dy = (y - curr_y)
        #     origin_x = 4 * 8
        #     origin_y = 4 * 8
        #
        #     if -4 <= dx <= 5 and -4 < dy <= 4:
        #         loc_x = (origin_x + dy * 8)
        #         loc_y = (origin_y + dx * 8)
        #         grayscale_downsampled_screen[loc_x: loc_x + 8, loc_y : loc_y + 8] *= self.target_symbol_mask

        self.screen_observation[:] = np.roll(self.screen_observation, self.screen_shape[0], axis=0)
        self.screen_observation[:self.screen_shape[0]] = grayscale_downsampled_screen

        return self.screen_observation

    def render(self):
        screen = self.screen.screen_ndarray()  # (144, 160, 3)

        return self.preprocess_screen(screen)

    def get_observed_stats(self):
        """
        We want to observe:
            - screen
            - exp
            - health fractions
            - badges
            - money

        :return: data
        """

        self.game_stats[MONEY].append(self.read_money())
        party_levels = self.read_party_levels()

        in_battle = self.read_in_battle()
        self.game_stats[IN_BATTLE].append(in_battle)
        self.game_stats[PARTY_LEVELS].append(party_levels)

        opp_level = self.read_opponent_level()
        # in_battle = opp_level not in (0, 255)
        if in_battle :
             self.latest_opp_level = np.maximum(opp_level, 1)

        self.game_stats[LEVEL_FRAQ].append(self.latest_opp_level / (max(party_levels)+1e-8))
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
        #self.game_stats[EVENTS_TRIGGERED].append(event_flag_indices)
        # self.triggered_event_flags[event_flag_indices] = 1
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

        if self.step_count > 2 and (map_id not in self.visited_maps
            or
            self.game_stats[BADGE_SUM][-1] != self.game_stats[BADGE_SUM][-2]):

            self.go_explore.add_starting_point(self.game_stats)
        self.go_explore.update_stats(self.game_stats)

        tmp = self.visited_maps | {map_id}
        self.game_stats[MAPS_VISITED].append(len(tmp))

        pos = self.read_pos()
        curr_coords = pos + [map_id]
        self.game_stats[COORDINATES].append(curr_coords)

        if len(self.last_walked_coordinates) == 0 or curr_coords != self.last_walked_coordinates[-1]:
            self.last_walked_coordinates.append(curr_coords)

        # x, y = pos
        # x2, y2, _ = self.entrance_coords
        # self.game_stats[ENTRANCE_DELTA_POS].append([
        #     x2 - x, y2 - y
        # ])
        # self.game_stats[SURROUNDING_TILES_VISITATION].append([
        #     self.visited_coordinates[(x, y, map_id)],
        #     self.visited_coordinates[(x+1, y, map_id)],
        #     self.visited_coordinates[(x-1, y, map_id)],
        #     self.visited_coordinates[(x, y+1, map_id)],
        #     self.visited_coordinates[(x, y-1, map_id)],
        # ])
        # self.game_stats[NOVELTY_COUNT].append(self.distinct_frames_observed)



        idx = 0
        for getter in self.observed_stats_config:
            idx = getter(
                observation_history=self.game_stats,
                post_processed_array=self.observed_stats,
                index=idx
            )

        assert idx == self.additional_features_shape[0], (idx, self.additional_features_shape[0])

        return self.observed_stats

    def get_event_flags(self):
        return self.triggered_event_flags

    def get_coordinates(self):
        return np.array(self.game_stats[COORDINATES][-1][-1:], dtype=np.uint8)

    def get_moved(self):
        walked = False if self.step_count < 2 else self.game_stats[COORDINATES][-1] != self.game_stats[COORDINATES][-2]
        return np.array([walked], dtype=np.uint8)
    
    def get_allowed_actions(self):
        allowed_actions = np.ones(self.action_space.n, dtype=np.uint8)
        
        if self.stuck_count < 8 and len(self.last_walked_coordinates) > 1 and not self.game_stats[IN_BATTLE][-1]:
            # Does not handle map changes
            curr_x, curr_y, map_id = self.last_walked_coordinates[-1]
            forbidden_locations = self.last_walked_coordinates[-2: -1]
            #past_x, past_y, past_map = self.last_walked_coordinates[-2]

            if [curr_x, curr_y - 1, map_id] in forbidden_locations:
                allowed_actions[3] = 0
            if [curr_x, curr_y + 1, map_id] in forbidden_locations:
                allowed_actions[0] = 0
            if [curr_x - 1, curr_y, map_id] in forbidden_locations:
                allowed_actions[1] = 0
            if [curr_x + 1, curr_y, map_id] in forbidden_locations:
                allowed_actions[2] = 0

            # if curr_y - past_y == 1:
            #     allowed_actions[3] = 0
            # elif curr_y - past_y == -1:
            #     allowed_actions[0] = 0
            # elif curr_x - past_x == 1:
            #     allowed_actions[1] = 0
            # elif curr_x - past_x == -1:
            #     allowed_actions[2] = 0
            
        return allowed_actions

        
    def step(self, action):

        self.run_action_on_emulator(action)
        obs = self._get_obs()
        self.step_count += 1

        reward = self.get_game_state_reward()
        self.episode_reward += reward

        self.maximum_experience_in_party_so_far = np.maximum(
            self.game_stats[TOTAL_EXPERIENCE][-1],
            self.maximum_experience_in_party_so_far
        )

        done = self.step_count >= self.max_steps_noised
        if done:
            self.on_episode_end()

        return obs, reward, False, done, {}

    def add_video_frame(self):
        screen = self.screen.screen_ndarray().copy()
        # if self.step_count > 1:
        #     x, y, goal_map_id = self.current_goal
        #     curr_x, curr_y, curr_map_id = self.game_stats[COORDINATES][-1]
        #
        #     if goal_map_id == curr_map_id and not self.game_stats[IN_BATTLE][-1]:
        #         dx = (x - curr_x)
        #         dy = (y - curr_y)
        #         origin_x = 4 * 16
        #         origin_y = 4 * 16
        #
        #         if -4 <= dx <= 5 and -4 < dy <= 4:
        #             loc_x = (origin_x + dy * 16)
        #             loc_y = (origin_y + dx * 16)
        #             screen[loc_x: loc_x + 16, loc_y: loc_y + 16] *= self.target_symbol_mask_debug
        self.full_frame_writer.add_image(screen)

    def update_frame_knn_index(self, frame):

        # if self.get_levels_sum() >= 22 and not self.levels_satisfied:
        #     self.levels_satisfied = True
        #     self.base_explore = self.knn_index.get_current_count()
        #     self.init_knn()

        # We want to clip the bottom where text appears
        #clipped_frame = frame[:-22]
        frame = cv2.resize(
            frame, (frame.shape[1]//4, frame.shape[0]//4), interpolation=cv2.INTER_NEAREST
        )[:-5, :, np.newaxis]

        frame_vector = np.float32(frame.flatten())


        if self.step_count >= 2:
            # reset on badge get
            if (
                    self.game_stats[BADGE_SUM][-1] != self.game_stats[BADGE_SUM][-2]
            ):
                pass
                #self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vector, np.array([self.knn_index.get_current_count()])
            )
            self.distinct_frames_observed += 0
        elif self.game_stats[COORDINATES][-1][-1] != 40:

            labels, distances = self.knn_index.knn_query(frame_vector, k=1)
            distance = distances[0][0]

            if distance > self.similar_frame_dist:

                self.knn_index.add_items(
                    frame_vector, np.array([self.distinct_frames_observed % self.num_elements])
                )
                self.distinct_frames_observed += 1

                nearest = self.knn_index.get_items(labels[0])

                nearest = np.reshape(nearest, frame.shape)

                delta = np.abs(nearest - frame)

                if self.distinct_frames_observed > 300:
                    self.save_screenshot("novelty_frames", f"{self.distinct_frames_observed}_{self.worker_index}",
                                         image=delta)

                return int(self.distinct_frames_observed > 300)

        return 0

    def on_episode_end(self):
        if self.save_final_state:
            self.save_screenshot("final_states", f'frame_r{self.episode_reward:.4f}.jpeg')

        if self.save_video:
            self.full_frame_writer.close()

        self.game_stats[TOTAL_BLACKOUT].append(sum(self.game_stats[BLACKOUT]))
        self.reset_count += 1
        self.max_steps *= self.additional_steps_per_episode

    def get_game_state_reward(self):
        """
        proposed reward function:
            - knn (reset every important event, hopefully)
            - sum of cqrt(exp) over pokemons
            - badges

        Exploration should be motivated through knn (navigation)
        and going to areas with higher level pokemons (more exp)
        :return:
        """

        rewards = {} #"novelty": self.update_frame_knn_index(obs["screen"])}

        if self.step_count >= 4:
            curr_coords = tuple(self.game_stats[COORDINATES][-1])
            past_coords = tuple(self.game_stats[COORDINATES][-2])
            if not self.game_stats[IN_BATTLE][-1] and curr_coords == past_coords:
                self.stuck_count += 1
            else:
                self.stuck_count = 0

            #goal_reached = int(curr_coords == self.current_goal)

            # if goal_reached:
            #     self.task_timesteps = self.goal_task_timeout_steps

            # we gain more experience as game moves on:
            total_delta_exp = 0
            total_healing = 0
            highest_party_level = max(self.game_stats[PARTY_LEVELS][-1])

            level_fraq = self.latest_opp_level / highest_party_level
            if level_fraq < 0.5:
                overleveled_penaly = 1e-3
            else:
                overleveled_penaly = np.minimum(level_fraq, 1)


            if curr_coords not in self.pokemon_centers:

                for i in range(6):

                    total_delta_exp += overleveled_penaly * np.maximum(
                        (self.game_stats[PARTY_EXPERIENCE][-1][i]
                        - self.game_stats[PARTY_EXPERIENCE][-2][i]) * int(self.game_stats[PARTY_EXPERIENCE][-2][i] != 0.)
                        , 0.
                    ) / highest_party_level**2 # **3 encourage going further

            if not any(self.game_stats[BLACKOUT][-2:]) and curr_coords[-1] in self.pokemon_centers:
                for i in range(6):
                    # We need to make sure one of the pokemons wasnt KO and healed
                    total_healing += int(
                        self.game_stats[PARTY_HEALTH][-1][i]-self.game_stats[PARTY_HEALTH][-2][i]
                        > 0.05
                        and
                        self.game_stats[PARTY_HEALTH][-2][i] > 0
                    )

            # shopping

            # curr_balls, curr_potions = self.game_stats[ITEMS][-1]
            # past_balls, past_potions = self.game_stats[ITEMS][-2]
            # curr_money = self.game_stats[MONEY][-1]
            # past_money = self.game_stats[MONEY][-2]
            # db = np.maximum(curr_balls - past_balls, 0)
            # dp = np.maximum(curr_potions - past_potions, 0)
            #
            # if db > 0:
            #     mean_cost = int((curr_money-past_money) / db)
            #
            # elif dp > 0:


            rewards.update(**{
                BLACKOUT: self.game_stats[BLACKOUT][-1],
                BADGE_SUM: (
                    np.maximum(self.game_stats[BADGE_SUM][-1] - self.game_stats[BADGE_SUM][-2], 0.) * (self.game_stats[BADGE_SUM][-2] + 1)
                ),
                TOTAL_EXPERIENCE: total_delta_exp,
                SEEN_POKEMONS : (
                    np.maximum(self.game_stats[SEEN_POKEMONS][-1] - self.game_stats[SEEN_POKEMONS][-2],
                               0.) * (self.game_stats[SEEN_POKEMONS][-2])
                ),
                TOTAL_EVENTS_TRIGGERED: (
                        np.maximum(self.game_stats[TOTAL_EVENTS_TRIGGERED][-1]
                        - self.game_stats[TOTAL_EVENTS_TRIGGERED][-2], 0) * (self.game_stats[TOTAL_EVENTS_TRIGGERED][-2] - 10)
                ),

                MAPS_VISITED: (
                #     2 == self.game_stats[MAP_ID][-1]
                #     and
                #     2 not in self.visited_maps
                # )
                #                          + 25 *int(
                #     1 == self.game_stats[MAP_ID][-1]
                #     and
                #     1 not in self.visited_maps
                # ) + 1 *int(
                #     12 == self.game_stats[MAP_ID][-1]
                #     and
                #     12 not in self.visited_maps
                # )
                (self.game_stats[MAPS_VISITED][-1] - self.game_stats[MAPS_VISITED][-2]) * self.game_stats[MAPS_VISITED][-2]
                #* self.game_stats[MAPS_VISITED][-1]
                ),

                # reward optimized walks
                # COORDINATES + "_NEG": np.minimum(r_nav, 0.),
                # COORDINATES + "_POS": np.maximum(r_nav, 0.),

                # Punish if walked into wall
                # COORDINATES: int(len(self.last_walked_coordinates) > 1 and
                #     (curr_coords
                #     ==
                #     self.last_walked_coordinates[-2])
                #     and
                #     walked
                # ),

                PARTY_HEALTH: int(total_healing > 0),

                MONEY : int(self.game_stats[MONEY][-1] - self.game_stats[MONEY][-2] > 50
                            and
                            curr_coords[-1] not in self.poke_marts
                            ),

                #GOAL_TASK : int(goal_reached)
            })

            if total_healing > 0:
                self.save_screenshot("debug", "healing")

        self.visited_maps.add(self.game_stats[MAP_ID][-1])
        self.visited_maps.add(self.game_stats[MAP_ID][-1])

        # if walked:
        #     self.visited_coordinates[curr_coords] = np.minimum(
        #         self.visited_coordinates[curr_coords] + 1, 5
        #     )


        total_reward = 0
        for reward_name, reward in rewards.items():
            scaled_reward = reward * self.reward_function_config[reward_name]
            self.game_stats["reward_"+reward_name].append(scaled_reward)
            total_reward += scaled_reward

        # if total_reward == 0 and not walked:
        #     total_reward = -3e-5

        return total_reward

    def save_screenshot(self, folder, name, image=None):
        ss_dir = self.s_path / Path(folder)
        ss_dir.mkdir(exist_ok=True)

        curr_screen = self.screen.screen_ndarray()
        plt.imsave(
            ss_dir / Path(f'{name}_original.jpeg'),
            curr_screen
        )
        observed = image if image is not None else self.screen_observation
        plt.imsave(
            ss_dir / Path(f'{name}_observed.jpeg'),
            observed[:, :, 0],
            cmap="gray"
        )

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


