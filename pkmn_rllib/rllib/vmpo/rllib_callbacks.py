from collections import defaultdict
from typing import Dict, Tuple, Union, Optional

import cv2
import hnswlib
from ray.rllib import Policy, SampleBatch, BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, PolicyID

import numpy as np

from pkmn_env.red import PkmnRedEnv


class PokemonCallbacks(
    DefaultCallbacks
):
    """
    Log important game stats
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.knn_index = None
        self.width = None
        self.height = None
        self.height_cut = None
        self.similar_frame_dist = 1_000_000.



    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        sub_env: PkmnRedEnv = base_env.get_sub_environments()[env_index]

        if hasattr(episode, "custom_metrics"):

            episode.custom_metrics.update(
                **{metric: sub_env.game_stats[metric][-1] for metric in sub_env.LOGGABLE_VALUES}
            )

            episode.custom_metrics.update(
                distinct_frames_observed=sub_env.distinct_frames_observed
            )

            episode.custom_metrics.update(
                **{metric: sum(sub_env.game_stats[metric]) for metric in sub_env.game_stats if "reward" in metric}
            )

    # def on_postprocess_trajectory(
    #     self,
    #     *,
    #     worker: "RolloutWorker",
    #     episode: Episode,
    #     agent_id: AgentID,
    #     policy_id: PolicyID,
    #     policies: Dict[PolicyID, Policy],
    #     postprocessed_batch: SampleBatch,
    #     original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
    #     **kwargs,
    # ) -> None:
    #
    #         screen_data_batch = postprocessed_batch[SampleBatch.OBS]["screen"]
    #         if self.knn_index is None:
    #             self.width = screen_data_batch[1].shape[0]//4
    #
    #             self.height = screen_data_batch[0].shape[1]//4
    #             self.height_cut = - 5
    #             self.knn_index = hnswlib.Index(space='l2', dim=self.width*self.height)
    #             self.knn_index.init_index(
    #                 max_elements=20_000, ef_construction=200, M=16
    #             )
    #             self.knn_index.set_ef(200)
    #             self.knn_index.set_num_threads(1)
    #
    #
    #         idx_delta = 10
    #         last_added_idx = -idx_delta
    #         total
    #         for idx, screen in enumerate(screen_data_batch):
    #
    #             if idx - last_added_idx >= idx_delta:
    #                 screen = cv2.resize(
    #                     screen, (self.height, self.width), interpolation=cv2.INTER_NEAREST
    #                 )[:-self.height_cut]
    #
    #                 screen_flat = screen.flatten()[np.newaxis]
    #
    #                 if self.knn_index.get_current_count() == 0:
    #                     # if index is empty add current frame
    #                     self.knn_index.add_items(
    #                         screen_flat, np.array([self.knn_index.get_current_count()])
    #                     )
    #                 else:
    #
    #                     labels, distances = self.knn_index.knn_query(screen_flat, k=1)
    #                     distance = distances[0][0]
    #
    #                     if distance > self.similar_frame_dist:
    #
    #                         self.knn_index.add_items(
    #                             screen_flat, np.array([self.knn_index.get_current_count()])
    #                         )
    #                         last_added_idx = idx
    #
    #                         postprocessed_batch[SampleBatch.REWARDS][idx] += 1.
    #
    #
    #
    #
    #
    #
    #



