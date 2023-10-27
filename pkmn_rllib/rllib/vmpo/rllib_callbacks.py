from collections import defaultdict
from typing import Dict, Tuple, Union, Optional

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

