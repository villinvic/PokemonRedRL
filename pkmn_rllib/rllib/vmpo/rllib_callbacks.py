from collections import defaultdict
from typing import Dict, Tuple, Union, Optional

from ray.rllib import Policy, SampleBatch, BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, PolicyID

import numpy as np

from baselines.red_gym_env import RedGymEnv


class PokemonCallbacks(
    DefaultCallbacks
):
    """
    Log important game stats
    """

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

        sub_inv: RedGymEnv = base_env.get_sub_environments()[env_index]

        if hasattr(episode, "custom_metrics"):

            episode.custom_metrics.update(
                distinct_frames_observed=sub_inv.distinct_frames_observed,
                maximum_opponent_level=sub_inv.max_opponent_level,
                event_rewards=sub_inv.max_event_rew,
                level_reward=sub_inv.max_level_rew,
                total_levels=sub_inv.get_levels_sum(),
                total_heatlh=sub_inv.last_health,
                healing_rewards=sub_inv.total_healing_rew,
                blackouts=sub_inv.died_count,
                exploration=sub_inv.progress_reward["explore"],
                badges=sub_inv.get_badges(),
                pokemon_counts=sub_inv.read_m(0xD163)
            )


