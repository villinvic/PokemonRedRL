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

from python_utils.collections import DefaultOrderedDict


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
    MAP_ID = "map_id"
    MAPS_VISITED = "maps_visited"
    BLACKOUT = "blackout"
    BADGES = "badges"
    BADGE_SUM = "badge_sum"
    MONEY = "money"
    COORDINATES = "coordinates"
    PARTY_EXPERIENCE = "party_experience"
    TOTAL_EXPERIENCE = "total_experience"
    TOTAL_BLACKOUT = "total_blackouts"
    PARTY_LEVELS = "party_levels"
    TOTAL_LEVELS = "total_levels"
    PARTY_HEALTH = "party_health"
    PARTY_FILLS = "party_fills"
    SEEN_POKEMONS = "seen_pokemons"
    CAUGHT_POKEMONS = "caught_pokemons"
    EVENTS_TRIGGERED = "events_triggered"
    TOTAL_EVENTS_TRIGGERED = "total_events_triggered"
    ORIENTATION = "orientation"
    SURROUNDING_TILES_VISITATION = "surrounding_tiles_visitation"
    NOVELTY_COUNT = "novelty_count"
    DELTA_LEVEL = "party_delta_level"
    ENTRANCE_DELTA_POS = "entrance_delta_pos"
    SENT_OUT = "SENT_OUT"

    LOGGABLE_VALUES = (
        MAPS_VISITED,
        BADGE_SUM,
        MONEY,
        TOTAL_LEVELS,
        TOTAL_EXPERIENCE,
        SEEN_POKEMONS,
        CAUGHT_POKEMONS,
        TOTAL_BLACKOUT,
        TOTAL_EVENTS_TRIGGERED
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
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.save_video = config['save_video'] and self.worker_index == 1
        self.fast_video = config['fast_video'] and self.worker_index == 1
        self.additional_steps_per_episode = config['additional_steps_per_episode']

        self.save_final_state = config['save_final_state'] and self.worker_index == 1

        self.screen_shape = (72, 80)  # (48, 56)
        self.similar_frame_dist = config['sim_frame_dist']
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)

        self.metadata = {"render.modes": []}
        self.reward_range = (-np.inf, np.inf)

        self.pokemon_centers = {
            41, 58, 64, 68, 81, 89, 133, 141, 154, 171, 174, 182,
        }

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
                name=PkmnRedEnv.BADGES
            ),
            # VariableGetter(
            #     name=PkmnRedEnv.MONEY,
            #     scale=5e-7
            # ),
            VariableGetter(
                name=PkmnRedEnv.SEEN_POKEMONS,
                scale=1e-2
            ),
            VariableGetter(
                name=PkmnRedEnv.CAUGHT_POKEMONS,
                scale=0.1
            ),
            VariableGetter(
                dim=6,
                name=PkmnRedEnv.PARTY_HEALTH
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
                name=PkmnRedEnv.PARTY_LEVELS,
                scale=0.02,
            ),
            VariableGetter(
                name=PkmnRedEnv.DELTA_LEVEL,
                scale=0.1,
                post_process_fn=lambda x: np.clip(x, -2., 2.)
            ),
            VariableGetter(
                name=PkmnRedEnv.TOTAL_EVENTS_TRIGGERED,
                scale=0.02 #319
            ),
            VariableGetter(
                name=PkmnRedEnv.MAPS_VISITED,
                scale=0.05
            ),
            VariableGetter(
                dim=6,
                name=PkmnRedEnv.PARTY_FILLS,
            ),
            VariableGetter(
                dim=6,
                name=PkmnRedEnv.SENT_OUT
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
            PkmnRedEnv.BLACKOUT                 :   -0.3,
            PkmnRedEnv.SEEN_POKEMONS            :   0.,
            PkmnRedEnv.TOTAL_EXPERIENCE         :   10.,  # 0.5
            PkmnRedEnv.BADGE_SUM                :   100.,
            PkmnRedEnv.MAPS_VISITED             :   0., # 3.
            PkmnRedEnv.TOTAL_EVENTS_TRIGGERED   :   1.,
            PkmnRedEnv.COORDINATES              :   0,
            # PkmnRedEnv.COORDINATES + "_NEG"     :   0.003 * 0.9,
            # PkmnRedEnv.COORDINATES + "_POS"     :   0.003,
            PkmnRedEnv.PARTY_HEALTH             :   1.,

            # Additional

            # Not really novelty but ok, we have to work on that
            #"novelty"                           :   1.,  # 1e-3  #/ (self.similar_frame_dist)


        }

        self.additional_features_shape = (
            sum(v.dim for v in self.observed_stats_config),
        )
        self.observed_stats = np.zeros(self.additional_features_shape, dtype=np.float32)

        #self.triggered_event_flags = np.zeros((0xD886 - 0xD747) * 8, dtype=np.uint8)

        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=self.screen_shape + (1,), dtype=np.uint8),
            "stats": spaces.Box(low=-np.inf, high=np.inf, shape=self.additional_features_shape, dtype=np.float32),
            #"flags": spaces.Box(low=0, high=1, shape=(len(self.triggered_event_flags),), dtype=np.uint8),
        })

        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type='headless' if config['headless'] else 'SDL2',
            hide_window='--quiet' in sys.argv,
            disable_renderer=False #not (self.save_video or self.fast_video)
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
        self.visited_maps = {37, 38, 39}  # red (first and second floor) and blue houses
        self.visited_coordinates = defaultdict(lambda: 0)
        self.entrance_coords = (5, 3, 40)
        self.highest_opponent_level_so_far = 5.
        self.last_reward_dict = {}
        self.last_walked_coordinates = []
        self.full_frame_writer = None

        #self.init_knn()

        self.inited = 0

    def init_knn(self):
        clipped_shape = (self.screen_shape[0]//4-5, self.screen_shape[1]//4)
        self.knn_index = hnswlib.Index(space='l2', dim=np.prod(clipped_shape))
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=200, M=16
        )
        self.knn_index.set_ef(200)
        self.knn_index.set_num_threads(1)

    def _get_obs(self):

        return {
            "screen" :   self.render(),
            "stats"  :   self.get_observed_stats(),
            #"flags"  :   self.get_event_flags()
        }

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        walked = False
        for i in range(self.act_freq):
            # release action, so they are stateless
            if not walked:
                walked = self.read_walk_animation() > 0
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])

                if 3 < action < 6:
                    # release button
                    self.pyboy.send_input(self.release_button[action - 4])

                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

            if self.save_video and not self.fast_video:
                self.add_video_frame()

            self.pyboy.tick()

        if self.save_video and self.fast_video:
            self.add_video_frame()

        return walked


    def reset(self, options=None, seed=None):

        del self.game_stats

        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.game_stats = DefaultOrderedDict(list)
        #self.triggered_event_flags = np.zeros((0xD886 - 0xD747) * 8, dtype=np.uint8)
        self.last_reward_dict = {}

        #  We init only once now
        # self.init_knn()
        # self.distinct_frames_observed = 0

        self.step_count = 0
        self.maximum_experience_in_party_so_far = 0
        self.episode_reward = 0
        self.visited_maps = {37, 38, 39}
        self.entrance_coords = (5, 30, 40)
        self.highest_opponent_level_so_far = 5.
        self.visited_coordinates = defaultdict(lambda: 0)
        self.last_walked_coordinates = []


        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir. ir(exist_ok=True)
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

        return np.uint8(grayscale_downsampled_screen)

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

        self.game_stats[PkmnRedEnv.MONEY].append(self.read_money())
        party_levels = self.read_party_levels()
        self.highest_opponent_level_so_far = np.maximum(self.highest_opponent_level_so_far, self.read_opponent_level())
        self.game_stats[PkmnRedEnv.PARTY_LEVELS].append(party_levels)

        self.game_stats[PkmnRedEnv.DELTA_LEVEL].append(max(party_levels) - self.highest_opponent_level_so_far)
        self.game_stats[PkmnRedEnv.TOTAL_LEVELS].append(sum(party_levels))
        party_experience = self.read_party_experience()
        self.game_stats[PkmnRedEnv.PARTY_EXPERIENCE].append(party_experience)
        self.game_stats[PkmnRedEnv.TOTAL_EXPERIENCE].append(sum(party_experience))
        badges = self.read_badges()
        self.game_stats[PkmnRedEnv.BADGES].append(badges)
        self.game_stats[PkmnRedEnv.BADGE_SUM].append(sum(badges))
        self.game_stats[PkmnRedEnv.SEEN_POKEMONS].append(self.read_seen())
        self.game_stats[PkmnRedEnv.CAUGHT_POKEMONS].append(self.read_caught())
        event_flag_indices = self.read_extensive_events()
        self.game_stats[PkmnRedEnv.TOTAL_EVENTS_TRIGGERED].append(len(event_flag_indices))
        self.game_stats[PkmnRedEnv.EVENTS_TRIGGERED].append(event_flag_indices)
        # self.triggered_event_flags[event_flag_indices] = 1
        self.game_stats[PkmnRedEnv.PARTY_FILLS].append(self.read_party_fills())
        self.game_stats[PkmnRedEnv.SENT_OUT].append(self.read_sent_out())
        party_health = self.read_party_health()
        self.game_stats[PkmnRedEnv.BLACKOUT].append(
            int(sum(party_health) == 0)
            and
            (len(self.game_stats[PkmnRedEnv.PARTY_HEALTH]) == 0 or sum(self.game_stats[PkmnRedEnv.PARTY_HEALTH][-1]) > 0)
        )
        self.game_stats[PkmnRedEnv.PARTY_HEALTH].append(party_health)

        map_id = self.read_map_id()
        self.game_stats[PkmnRedEnv.MAP_ID].append(map_id)

        tmp = self.visited_maps | {map_id}
        self.game_stats[PkmnRedEnv.MAPS_VISITED].append(len(tmp))

        pos = self.read_pos()

        self.game_stats[PkmnRedEnv.COORDINATES].append(pos + [map_id])

        # x, y = pos
        # x2, y2, _ = self.entrance_coords
        # self.game_stats[PkmnRedEnv.ENTRANCE_DELTA_POS].append([
        #     x2 - x, y2 - y
        # ])
        # self.game_stats[PkmnRedEnv.SURROUNDING_TILES_VISITATION].append([
        #     self.visited_coordinates[(x, y, map_id)],
        #     self.visited_coordinates[(x+1, y, map_id)],
        #     self.visited_coordinates[(x-1, y, map_id)],
        #     self.visited_coordinates[(x, y+1, map_id)],
        #     self.visited_coordinates[(x, y-1, map_id)],
        # ])
        # self.game_stats[PkmnRedEnv.NOVELTY_COUNT].append(self.distinct_frames_observed)

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

    def step(self, action):

        walked = self.run_action_on_emulator(action)
        obs = self._get_obs()
        self.step_count += 1

        reward = self.get_game_state_reward(obs, walked)
        self.episode_reward += reward

        self.maximum_experience_in_party_so_far = np.maximum(
            self.game_stats[PkmnRedEnv.TOTAL_EXPERIENCE][-1],
            self.maximum_experience_in_party_so_far
        )

        done = self.step_count >= self.max_steps_noised
        if done:
            self.on_episode_end()

        return obs, reward, False, done, {}

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.screen.screen_ndarray())

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
                    self.game_stats[PkmnRedEnv.BADGE_SUM][-1] != self.game_stats[PkmnRedEnv.BADGE_SUM][-2]
            ):
                pass
                #self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vector, np.array([self.knn_index.get_current_count()])
            )
            self.distinct_frames_observed += 0
        elif self.game_stats[PkmnRedEnv.COORDINATES][-1][-1] != 40:

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

        self.game_stats[PkmnRedEnv.TOTAL_BLACKOUT].append(sum(self.game_stats[PkmnRedEnv.BLACKOUT]))
        self.reset_count += 1
        self.max_steps *= self.additional_steps_per_episode

    def get_game_state_reward(self, obs, walked):
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
            curr_coords = tuple(self.game_stats[PkmnRedEnv.COORDINATES][-1])
            # past_coords = tuple(self.game_stats[PkmnRedEnv.COORDINATES][-2])
            # past_2_coords = tuple(self.game_stats[PkmnRedEnv.COORDINATES][-3])
            # past_3_coords = tuple(self.game_stats[PkmnRedEnv.COORDINATES][-4])
            if walked:
                self.last_walked_coordinates.append(curr_coords)
            #
            # if (
            #         self.entrance_coords is None
            #         or
            #         (
            #             self.entrance_coords[-1] != curr_coords[-1]
            #             and
            #             curr_coords[-1] == past_coords[-1] == past_2_coords[-1]
            #         )
            # ):
            #     self.entrance_coords = curr_coords
            #
            # if (
            #         curr_coords[-1] == past_coords[-1] == past_3_coords[-1] == self.entrance_coords[-1]
            # ):
            #     dx = abs(curr_coords[0] - self.entrance_coords[0])
            #     dy = abs(curr_coords[1] - self.entrance_coords[1])
            #     # the past coord might be far away if we teleported, but should not happen as we still have to walk away
            #     # from the entrance coord a bit before getting there
            #     dx2 = abs(past_coords[0] - self.entrance_coords[0])
            #     dy2 = abs(past_coords[1] - self.entrance_coords[1])
            #
            #     assert abs(dx-dx2) + abs(dy-dy2) <= 1, self.game_stats[PkmnRedEnv.COORDINATES][-6:]
            #
            #     r_nav = dx - dx2 + dy - dy2
            #
            #     # if dx < 9 or dy < 9: # we do not reward for navigating in small rooms
            #     #     r_nav = np.minimum(r_nav, 0.)
            #
            # elif self.entrance_coords[-1] != curr_coords[-1]:
            #     r_nav = -1.
            # else:
            #     r_nav = 0.

            # we gain more experience as game moves on:
            total_delta_exp = 0
            total_healing = 0
            level_fraction = np.minimum(1., self.highest_opponent_level_so_far/max(self.game_stats[PkmnRedEnv.PARTY_LEVELS][-1]))
            for i in range(6):
                # Can be hacked with pc, let's see :)

                total_delta_exp += level_fraction * np.maximum(
                    (self.game_stats[PkmnRedEnv.PARTY_EXPERIENCE][-1][i]
                    - self.game_stats[PkmnRedEnv.PARTY_EXPERIENCE][-2][i]) * int(self.game_stats[PkmnRedEnv.PARTY_EXPERIENCE][-2][i] != 0.)
                    , 0.
                ) / np.maximum(max(self.game_stats[PkmnRedEnv.PARTY_LEVELS][-1])**3,
                6.)

            if not any(self.game_stats[PkmnRedEnv.BLACKOUT][-2:]) and curr_coords[-1] in self.pokemon_centers:
                for i in range(6):
                    # We need to make sure one of the pokemons wasnt KO and healed
                    total_healing += int(
                        self.game_stats[PkmnRedEnv.PARTY_HEALTH][-1][i]-self.game_stats[PkmnRedEnv.PARTY_HEALTH][-2][i]
                        > 0.05
                        and
                        self.game_stats[PkmnRedEnv.PARTY_HEALTH][-2][i] > 0
                    )

            rewards.update(**{
                PkmnRedEnv.BLACKOUT: self.game_stats[PkmnRedEnv.BLACKOUT][-1],
                PkmnRedEnv.BADGE_SUM: (
                    np.maximum(self.game_stats[PkmnRedEnv.BADGE_SUM][-1] - self.game_stats[PkmnRedEnv.BADGE_SUM][-2], 0.)
                ),
                PkmnRedEnv.TOTAL_EXPERIENCE: total_delta_exp,
                PkmnRedEnv.SEEN_POKEMONS : (
                    np.maximum(self.game_stats[PkmnRedEnv.SEEN_POKEMONS][-1] - self.game_stats[PkmnRedEnv.SEEN_POKEMONS][-2],
                               0.)
                ),
                PkmnRedEnv.TOTAL_EVENTS_TRIGGERED: (
                        self.game_stats[PkmnRedEnv.TOTAL_EVENTS_TRIGGERED][-1]
                        - self.game_stats[PkmnRedEnv.TOTAL_EVENTS_TRIGGERED][-2]
                      #                       * (self.game_stats[PkmnRedEnv.EVENTS_TRIGGERED][-1] - 11
                ),

                PkmnRedEnv.MAPS_VISITED: (
                #     2 == self.game_stats[PkmnRedEnv.MAP_ID][-1]
                #     and
                #     2 not in self.visited_maps
                # )
                #                          + 25 *int(
                #     1 == self.game_stats[PkmnRedEnv.MAP_ID][-1]
                #     and
                #     1 not in self.visited_maps
                # ) + 1 *int(
                #     12 == self.game_stats[PkmnRedEnv.MAP_ID][-1]
                #     and
                #     12 not in self.visited_maps
                # )
                (self.game_stats[PkmnRedEnv.MAPS_VISITED][-1] - self.game_stats[PkmnRedEnv.MAPS_VISITED][-2])
                #* self.game_stats[PkmnRedEnv.MAPS_VISITED][-1]
                ),

                # reward optimized walks
                # PkmnRedEnv.COORDINATES + "_NEG": np.minimum(r_nav, 0.),
                # PkmnRedEnv.COORDINATES + "_POS": np.maximum(r_nav, 0.),

                PkmnRedEnv.COORDINATES: int(
                    (curr_coords
                    in
                    self.last_walked_coordinates[-6:-1])
                    and
                    walked
                ),

                PkmnRedEnv.PARTY_HEALTH: int(total_healing > 0)
            })

            if total_healing > 0:
                self.save_screenshot("debug", "healing")

        self.visited_maps.add(self.game_stats[PkmnRedEnv.MAP_ID][-1])
        self.visited_maps.add(self.game_stats[PkmnRedEnv.MAP_ID][-1])

        # if walked:
        #     self.visited_coordinates[curr_coords] = np.minimum(
        #         self.visited_coordinates[curr_coords] + 1, 5
        #     )


        total_reward = 0
        for reward_name, reward in rewards.items():
            scaled_reward = reward * self.reward_function_config[reward_name]
            self.game_stats["reward_"+reward_name].append(scaled_reward)
            total_reward += scaled_reward

        if total_reward == 0 and not walked:
            total_reward = -3e-5

        return total_reward

    def save_screenshot(self, folder, name, image=None):
        ss_dir = self.s_path / Path(folder)
        ss_dir.mkdir(exist_ok=True)

        curr_screen = self.screen.screen_ndarray()
        plt.imsave(
            ss_dir / Path(f'{name}_original.jpeg'),
            curr_screen
        )
        observed = image if image is not None else self.preprocess_screen(curr_screen)
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
        return (100 * 100 * self.read_bcd(self.read_m(0xD347)) +
                100 * self.read_bcd(self.read_m(0xD348)) +
                self.read_bcd(self.read_m(0xD349)))

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
        return self.read_m(0xCFE8)

    def read_sent_out(self) -> List:
        return [int(i == self.read_m(0xCC2F)) for i in range(6)]


