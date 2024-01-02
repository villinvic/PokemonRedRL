import pickle
import queue
import uuid
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Optional, Tuple, Union, List

import Levenshtein
import gymnasium
import numpy as np
import ray
from gymnasium.core import ObsType, ActType, RenderFrame
import multiprocessing as mp

from pkmn_env.enums import BADGE_SUM, CAUGHT_POKEMONS, SEEN_POKEMONS
from pkmn_env.red_no_render import PkmnRedEnvNoRender

"""
Async algo

TODO :
      Possible operations:
          - insertion
          - deletion
          - mutation
          - multi-point crossover
      fitness:
          - sequence length
          - num badges
          - num events
          - total_exp_scaled
          - total money
          - pokemon seen
          - pokemon caught entropy
          - novelty (distance)
          
"""


class ActionSequence:

    def __init__(
            self,
            action_space: gymnasium.Space,
            config,
    ):
        self.config = config
        self.curr_action_idx = 0
        self.seq_len = 0
        self.n_actions = action_space.n
        self.ending_action = action_space.n
        self.action_sequence_length_limits = config["action_sequence_limits"]
        self.sequence = np.full((self.action_sequence_length_limits[1]), dtype=np.uint8, fill_value=self.ending_action)

    def distance(self, other: "ActionSequence"):
        return Levenshtein.distance(self.sequence, other.sequence)

    def initialize_randomly(self):
        self.seq_len = self.action_sequence_length_limits[0]

        self.sequence[:  self.seq_len] = np.random.randint(0, self.n_actions, self.seq_len)
        self.sequence[self.seq_len] = self.ending_action

    def validate_sequence(self):
        # TODO : speedup
        sequence = list(self.sequence)

        return sequence

    def __iter__(self):
        self.curr_action_idx = 0
        return self

    def __next__(self) -> ActType:
        if self.curr_action_idx < len(self.sequence) and self.sequence[self.curr_action_idx] != self.ending_action:
            action = self.sequence[self.curr_action_idx]
            self.curr_action_idx += 1
            return action
        else:
            raise StopIteration

    def update(self, other: "ActionSequence"):
        self.sequence[:] = other.sequence
        self.seq_len = other.seq_len
        self.curr_action_idx = 0

    def mutate(self):
        old_sequence = self.sequence.copy()
        # random addition and removals of subsequences
        idx_delta = 0
        new_seq_len = self.seq_len
        for idx in range(self.seq_len):
            if np.random.random() < self.config["subsequence_mutation_rate"]:
                # copy a subsequence and insert it anywhere before or after that subsequence

                if np.random.random() < 0.5:
                    # Insertion
                    upper = np.minimum(1 + self.config["max_subsequence_length"],
                                                             1 + self.action_sequence_length_limits[1] - new_seq_len)

                    if upper == 0:
                        continue

                    length = np.random.randint(0, upper)



                    copy_idx_start = np.random.randint(0, self.seq_len - length)
                    copy_idx_end = copy_idx_start + length

                    self.sequence[:] = np.concatenate([self.sequence[:idx], old_sequence[copy_idx_start:copy_idx_end],
                                                       self.sequence[idx:-length]])
                    new_seq_len += length
                else:
                    # Removal
                    upper = np.minimum(1 + self.config["max_subsequence_length"],
                                                             new_seq_len - self.action_sequence_length_limits[0])
                    if upper == 0:
                        continue
                    length = np.random.randint(0, upper)

                    self.sequence[:] = np.concatenate([self.sequence[:idx], self.sequence[idx + length:],
                                                       np.full((length,), fill_value=self.ending_action)])

                    new_seq_len -= length

        self.seq_len = new_seq_len

        mutation_indices = np.random.random(self.seq_len) < self.config["mutation_rate"]
        self.sequence[:self.seq_len][mutation_indices] = np.random.randint(0, self.n_actions, mutation_indices.sum())

    def crossover(self, other: "ActionSequence"):
        points = np.sort(np.random.choice(np.maximum(self.seq_len, other.seq_len),
                                          self.config["crossover_n_points"], replace=False))

        new_sequence = self.sequence.copy()

        prev_point = 0
        parents = [self.sequence, other.sequence]
        np.random.shuffle(parents)

        for i, next_point in enumerate(points):
            parent = parents[i % 2]
            new_sequence[prev_point: next_point] = parent[prev_point: next_point]
            prev_point = next_point

        parent = parents[(len(points) + 1) % 2]

        new_sequence[prev_point:] = parent[prev_point:]

        self.sequence[:] = new_sequence

        self.seq_len = np.min(np.argwhere(self.sequence == self.ending_action))


class Individual:
    def __init__(
            self,
            environment,
            config,
            age,
    ):
        self.ID = None  # We should set it manually after
        self._action_sequence = ActionSequence(environment.action_space, config)
        self.config = config
        self.age = age

        self.evaluation_dict = {}

    @property
    def action_sequence(self):
        return self._action_sequence

    @action_sequence.setter
    def action_sequence(self, sequence: ActionSequence):
        self.evaluation_dict = {}
        self.ID = None
        self._action_sequence.update(sequence)

    def eval(self, environment_instance: gymnasium.Env):
        environment_instance.reset()

        # Run action sequence
        times = []

        t = time()
        for action in self.action_sequence:
            t2 = time()
            times.append(t2 - t)
            t = t2

            environment_instance.step(action)

        print(self.ID, "action computation stats", np.max(times), np.mean(times), np.min(times))

        self.evaluation_dict = environment_instance.get_stats()

        # Identifies evaluation dicts
        self.evaluation_dict["GA/ID"] = self.ID

        return self

    def initialize_randomly(self, ID):
        self.action_sequence.initialize_randomly()
        self.ID = ID
        return self

    def set_as(self, individual: "Individual"):
        self.action_sequence = individual.action_sequence
        self.evaluation_dict = individual.evaluation_dict
        self.age = individual.age

    def __repr__(self):
        return f"<{self.ID}={self.evaluation_dict}({self.action_sequence.seq_len})>"

    def evolve(self, new_id, parents):
        self.set_as(parents[0])
        self.action_sequence.crossover(parents[1].action_sequence)
        self.action_sequence.mutate()
        self.evaluation_dict = {}
        self.ID = new_id

    def pairwise_novelty(self, other: "Individual"):
        return self.action_sequence.distance(other.action_sequence)

class Worker:
    def __init__(self, worker_id, environment_cls, config):
        self.worker_id = worker_id
        self.environment = environment_cls(config["env_config"])
        self.config = config

    @classmethod
    def as_remote(cls):
        return ray.remote(
            num_cpus=4,
            num_gpus=0,
        )(cls)

    def eval(self, individual: Individual):
        return self.worker_id, individual.eval(self.environment)


class Population:

    def __init__(self, environment: gymnasium.Env, config: dict):
        self.curr_individual_idx = 0

        self.environment = environment
        self.config = config
        self.size = config["population_size"]
        self.age = 0

        self.population = defaultdict(
            lambda: Individual(
                environment=environment,
                config=config,
                age=self.age
            )
        )

    def initialize(self):
        for _ in range(self.size):
            ID = str(uuid.uuid4())[:8]
            self.population[ID].initialize_randomly(ID=ID)

    def __getitem__(self, item):
        return self.population[item]

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.items().__iter__()

    def to_evaluate(self):
        return [
            individual_id for individual_id, individual in self.population.items()
            if not individual.evaluation_dict
        ]

    def evaluated_population(self):

        return {
            i_id: individual for i_id, individual in self.population.items() if "GA/ID" in individual.evaluation_dict
        }

    def compute_fitnesses(self):
        evaluated_population = self.evaluated_population()
        for individual_id, individual in evaluated_population.items():
            other_ids = list(evaluated_population.keys())
            other_ids.remove(individual_id)
            min_distance = min([
                individual.pairwise_novelty(self.population[other_id])
                for other_id in np.random.choice(other_ids,
                                                 np.minimum(self.config["novelty_n_samples"],
                                                            len(evaluated_population) - 1),
                                                 replace=False)
            ])
            individual.evaluation_dict["novelty"] = min_distance
            individual.evaluation_dict["length"] = individual.action_sequence.seq_len

            individual.evaluation_dict["GA/FITNESS"] = sum([
                individual.evaluation_dict[metric] * scale for metric, scale in self.config["fitness_config"].items()
            ])
        return evaluated_population

    def ranking(self):
        rankable_individuals = self.compute_fitnesses()
        ranking = sorted(list(rankable_individuals.keys()),
                         key=lambda i_id: rankable_individuals[i_id].evaluation_dict["GA/FITNESS"])
        return ranking

    def select(self):
        ranking = self.ranking()
        to_drop = len(ranking) - self.size
        if to_drop > 0:
            for individual_id in ranking[:to_drop]:
                # drop the bottom 50%:
                self.population.pop(individual_id)

    def get_matting(self):
        available_for_matting = [i_id for i_id, individual in self.population.items() if
                                 "GA/FITNESS" in individual.evaluation_dict]

        fitnesses_exp = np.exp(
            np.array([self.population[i_id].evaluation_dict["GA/FITNESS"] for i_id in available_for_matting]))

        probs = fitnesses_exp / fitnesses_exp.sum()

        parents = np.random.choice(available_for_matting, 2, p=probs)

        return [self.population[p] for p in parents]

    def make_offspring(self):
        new_id = str(uuid.uuid4())[:8]
        self.population[new_id].evolve(new_id, parents=self.get_matting())
        return new_id

    def save_individual(self, ID, path=""):

        with open(path, "wb+") as f:
            pickle.dump(self.population[ID].action_sequence.sequence, f)

    def __repr__(self):
        return str(list(self.population.keys()))


class Archive(Population):
    def __init__(self, environment: gymnasium.Env, config: dict, max_size=2048):
        super().__init__(environment, config)
        self.max_entries = max_size

    def __getitem__(self, individual: Individual):
        if len(self.population) == self.max_entries:
            print("Archive full, TBA")
            return None
        else:
            assert individual.ID is None, "Non-initialized individual added to archive !"
            self.population[individual.ID].set_as(individual)

    def __len__(self):
        return len(self.population)


class GA:
    def __init__(self, env_cls, config):
        base_env = env_cls(config["env_config"])
        self.env_cls = env_cls
        self.population = Population(base_env, config)
        #self.eval_workers = {w_id: Worker.as_remote().remote(w_id, env_cls, config) for w_id in range(config["num_workers"])}
        self.eval_workers = mp.Pool(config["num_workers"])
        self.available_worker_ids = {w_id for w_id in range(config["num_workers"])}

        self.to_eval_queue = []

        self.num_expected_evals = len(self.population)

        self.runner = self.main_loop()

    def initialize_population(self):
        self.population.initialize()

    def main_loop(self):
        jobs = []
        while True:

            done_jobs = []
            evaluated_individuals = []
            jobs_to_send = self.num_expected_evals
            jobs_sent = []
            while len(done_jobs) < self.num_expected_evals:
                if jobs_to_send > len(jobs_sent):
                    # TODO : sends more than one job there !
                    new_jobs = self.send_job_to_available_workers()
                    jobs_sent.extend(new_jobs)
                    jobs.extend(new_jobs)

                res = []
                for job in jobs:
                    try:
                        r = job.get(block=False)
                        res.append(r)
                    except queue.Empty as e:
                        pass

                    jobs.remove(job)

                    done_jobs.append(job)

                # latest_done_jobs, jobs = ray.wait(
                #     jobs,
                #     num_returns=1,
                #     timeout=None,
                #     fetch_local=False
                # )
                if len(res) > 0:
                    print("done jobs:", len(done_jobs))
                    done_workers = []
                    for w_id, eval_dict in res:
                        evaluated_individuals.append(eval_dict)
                        done_workers.append(w_id)

                    self.available_worker_ids |= set(done_workers)

            yield evaluated_individuals

    def get_next_individual_id(self) -> Union[str, None]:
        if len(self.to_eval_queue) == 0:
            return None
        else:
            return self.to_eval_queue.pop(0)

    def send_job_to_available_workers(self, max_jobs=1024):
        jobs = []
        while self.available_worker_ids:

            next_individual_id = self.get_next_individual_id()

            if next_individual_id is None:
                break

            w_id = self.available_worker_ids.pop()

            # print("job sent :", w_id, next_individual_id)
            # print("available workers:", self.available_worker_ids)
            # print("to eval:", self.to_eval_queue)
            def eval_individual():
                self.population[next_individual_id].eval(self.env_cls)

            jobs.append(self.eval_workers.apply_aync(eval_individual, ()))
            # jobs.append(
            #     self.eval_workers[w_id].eval.remote(
            #         self.population[next_individual_id]
            #     ))

            if len(jobs) == max_jobs:
                break

        return jobs

    def __call__(self):

        while True:

            for w_id in self.available_worker_ids:
                new_id = self.population.make_offspring()
                self.to_eval_queue.append(new_id)

            if len(self.population.evaluated_population()) >= 2 * self.population.size:
                self.population.select()

                evaluated = self.population.evaluated_population()
                print("Best individual so far:")
                best = sorted(evaluated, key=lambda i_id: -evaluated[i_id].evaluation_dict["GA/FITNESS"])[0]
                self.population.save_individual(best, "best_individual.pkl")
                print(self.population[best])

            for evaluated_individual in next(self.runner):
                self.population[evaluated_individual.ID].evaluation_dict = evaluated_individual.evaluation_dict

    def init_eval(self):
        self.to_eval_queue = list(self.population.population.keys())
        self.num_expected_evals = len(self.population)

        for evaluated_individual in next(self.runner):
            self.population[evaluated_individual.ID].evaluation_dict = evaluated_individual.evaluation_dict

        self.num_expected_evals = 1
        self.population.compute_fitnesses()


if __name__ == '__main__':

    class TestEnv(gymnasium.Env):

        def __init__(self, config):
            self.state = 0

            self.action_space = gymnasium.spaces.Discrete(4)
            self.observation_space = gymnasium.spaces.Discrete(100)

        def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict] = None,
        ) -> Tuple[ObsType, dict]:
            self.state = 0
            return 0, {}

        def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
            self.state += action

            done = self.state >= 100
            self.state = np.minimum(self.state, 100)

            return self.state, self.state, done, done, {}

        def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
            pass

        def emulate_action_sequence(self, action_sequence: ActionSequence):
            total_score = 0
            for action in action_sequence:
                _, r, _, done, _ = self.step(action)

                total_score += r

            evaluation = self.get_episode_evaluation()
            evaluation.update(total_score=total_score)

            return evaluation

        def get_episode_evaluation(self):
            return {}


    config = {
        "action_sequence_limits"   : (256, 2048*4),
        "env_config"               : {
            "init_state"  : "deepred_post_parcel_pokeballs",
            "session_path": Path("sessions/tests"),
            "gb_path"     : "pokered.gbc",
            "render"      : False
        },
        "population_size"          : 6,
        "num_workers"              : 6,
        "fitness_config"           : {
            "episode_reward": 10.,
            BADGE_SUM       : 100.,

            CAUGHT_POKEMONS : 0.5,
            SEEN_POKEMONS   : 0.1,
            "novelty"       : 1e-5,
            "length"        : -1e-4,

        },
        "novelty_n_samples"        : 64,
        "crossover_n_points"       : 4,
        "mutation_rate"            : 0.05,
        "subsequence_mutation_rate": 1e-3,
        "max_subsequence_length"   : 16
    }

    ray.init()

    runner = GA(env_cls=PkmnRedEnvNoRender, config=config)
    runner.initialize_population()
    runner.init_eval()
    print("init done")
    runner()
