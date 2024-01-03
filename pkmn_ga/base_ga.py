import os
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

from pkmn_env.enums import BADGE_SUM, CAUGHT_POKEMONS, SEEN_POKEMONS, MAP_ID, TOTAL_EVENTS_TRIGGERED, MAPS_VISITED
from pkmn_env.go_explore import GoExplorePokemon
from pkmn_env.red_no_render import PkmnRedEnvNoRender
from python_utils.collections import DefaultOrderedDict

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
        self.mutable_start = 0
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
        self.curr_action_idx = self.mutable_start
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
        self.mutable_start = other.mutable_start

    def mutate(self):
        # old_sequence = self.sequence.copy()[self.mutable_start:]
        # # random addition and removals of subsequences
        # idx_delta = 0
        # new_seq_len = self.seq_len
        # for idx in range(self.mutable_start, self.seq_len):
        #     if np.random.random() < self.config["subsequence_mutation_rate"]:
        #         # copy a subsequence and insert it anywhere before or after that subsequence
        #
        #         if np.random.random() < 0.5:
        #             # Insertion
        #             upper = np.minimum(1 + self.config["max_subsequence_length"],
        #                                                      1 + self.action_sequence_length_limits[1] - new_seq_len)
        #
        #             if upper == 0:
        #                 continue
        #
        #             length = np.random.randint(1, upper+1)
        #
        #             copy_idx_start = np.random.randint(0, self.seq_len - length)
        #             copy_idx_end = copy_idx_start + length
        #
        #             self.sequence[self.mutable_start:] = np.concatenate([self.sequence[self.mutable_start:idx], old_sequence[copy_idx_start:copy_idx_end],
        #                                                self.sequence[idx:-length]])
        #             new_seq_len += length
        #         else:
        #             # Removal
        #             upper = np.minimum(1 + self.config["max_subsequence_length"],
        #                                                      new_seq_len - self.action_sequence_length_limits[0])
        #             if upper == 0:
        #                 continue
        #             length = np.random.randint(1, upper+1)
        #
        #             print(len(self.sequence[self.mutable_start:idx]), len(self.sequence[idx + length:]), length)
        #             self.sequence[self.mutable_start:] = np.concatenate([self.sequence[self.mutable_start:idx], self.sequence[idx + length:],
        #                                                np.full((length,), fill_value=self.ending_action)])
        #
        #             new_seq_len -= length
        #
        # self.seq_len = new_seq_len

        new_sequence = []
        new_seq_len = self.seq_len

        mutation_rate = self.config["mutation_rate"]
        mutation_prob = mutation_rate/3
        mutation_func = lambda : np.random.choice(4, p=[1-mutation_rate, mutation_prob, mutation_prob, mutation_prob])

        for action in self.sequence[self.mutable_start:self.seq_len]:
            mutation_type = mutation_func()
            if mutation_type == 1:
                # mutate_action
                new_sequence.append(np.random.randint(self.ending_action))
            elif mutation_type == 2 and new_seq_len < self.action_sequence_length_limits[1]:
                # add action
                new_sequence.append(np.random.randint(self.ending_action))
                new_sequence.append(action)
                new_seq_len += 1
            elif mutation_type == 3 and new_seq_len > self.action_sequence_length_limits[0]:
                # forget action
                new_seq_len -= 1
            else:
                # do nothing:
                new_sequence.append(action)

        self.sequence[self.mutable_start:self.mutable_start+len(new_sequence)] = new_sequence

        self.seq_len = new_seq_len
        self.sequence[new_seq_len:] = self.ending_action

        # mutation_indices = np.random.random(self.seq_len-self.mutable_start) < self.config["mutation_rate"]
        # self.sequence[self.mutable_start:self.seq_len][mutation_indices] = np.random.randint(0, self.n_actions, mutation_indices.sum())

    def crossover(self, other: "ActionSequence"):
        if self.mutable_start == other.mutable_start and (
                np.maximum(self.seq_len, other.seq_len) - self.mutable_start > self.config["crossover_n_points"]
        ):
            points = np.sort(np.random.choice(np.maximum(self.seq_len, other.seq_len) - self.mutable_start,
                                              self.config["crossover_n_points"], replace=False)) + self.mutable_start

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

            self.sequence[self.mutable_start:] = new_sequence[self.mutable_start:]

            if self.ending_action in self.sequence:

                self.seq_len = np.min(np.argwhere(self.sequence == self.ending_action))
            else:
                self.seq_len = self.action_sequence_length_limits[1]
        else:
            # Retain first parent
            pass

    def set_base(self, base):
        max_length = self.action_sequence_length_limits[1]

        self.sequence[:len(base)] = base

        if self.mutable_start < len(base):
            addition = np.minimum(np.random.randint(32, 512), max_length-len(base))
            if addition == 0:
                raise Exception
            self.sequence[len(base):len(base)+addition] = np.random.randint(0, self.n_actions, addition)
            new_seq_len = len(base) + addition
        else:
            new_seq_len = self.seq_len

        self.seq_len = new_seq_len
        self.mutable_start = len(base)



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
        self.start_point = {}
        self.end_point = {}
        self.evaluation_dict = {}

    @property
    def action_sequence(self):
        return self._action_sequence

    @action_sequence.setter
    def action_sequence(self, sequence: ActionSequence):
        self.evaluation_dict = {}
        self.ID = None
        self._action_sequence.update(sequence)

    def eval(self, environment_instance: gymnasium.Env,):

        environment_instance.reset(options=self.start_point)

        # Run action sequence
        times = []

        #t = time()
        go_explore_seen_counts = defaultdict(lambda: 0)
        key_states = defaultdict(lambda: {
            "cost": np.inf,
            "stats": {},
            "game_state": None
        })

        if "step_count" in self.start_point:
            assert self.action_sequence.mutable_start == self.start_point["step_count"], (
                    self.action_sequence.mutable_start, self.start_point
            )

        for action in self.action_sequence:
            idx = self.action_sequence.curr_action_idx
            # t2 = time()
            # times.append(t2 - t)
            # t = t2
            #print(worker_id, times[-1])
            environment_instance.step(action)
            identifier = tuple(environment_instance.game_stats[feature][-1]
                               for feature in self.config["go_explore_relevant_features"])
            go_explore_seen_counts[identifier] += 1

            if identifier not in key_states:
                key_states[identifier]["cost"] = idx
                key_states[identifier]["stats"] = environment_instance.get_stats()
                key_states[identifier]["game_state"] = environment_instance.game_state()

        if len(key_states) == 0:
            print(self.action_sequence.sequence[self.action_sequence.mutable_start:], self.action_sequence.mutable_start)

        #print(self.ID, "action computation stats", np.max(times), np.mean(times), np.min(times))

        self.evaluation_dict = environment_instance.get_stats()
        self.evaluation_dict["GO_EXPLORE/" + GoExplorePokemon.TIMES_SEEN] = go_explore_seen_counts
        self.evaluation_dict["GO_EXPLORE/key_states"] = key_states


        self.end_point = environment_instance.game_state()


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

        self.start_point = individual.start_point
        self.end_point = individual.end_point

    def __repr__(self):
        return f"<{self.ID}={self.evaluation_dict}({self.action_sequence.seq_len})>"

    def evolve(self, new_id, parents, archive):
        self.set_as(parents[0])

        self.build_from_base(archive.sample_starting_point())

        #self.action_sequence.crossover(parents[1].action_sequence)
        self.action_sequence.mutate()
        self.evaluation_dict = {}
        self.ID = new_id

    def pairwise_novelty(self, other: "Individual"):
        return self.action_sequence.distance(other.action_sequence)

    def build_from_base(self, base: dict):

        self.start_point = base["start_point"]
        self.end_point = {}
        assert len(base["action_sequence"]) == base["start_point"]["step_count"], base
        self.action_sequence.set_base(base["action_sequence"])

class Worker:
    def __init__(self, worker_id, environment_cls, config):
        self.worker_id = worker_id
        self.environment = environment_cls(config["env_config"])
        self.config = config

    @classmethod
    def as_remote(cls):
        return ray.remote(
            num_cpus=1,
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

    def compute_true_fitness(self, stats):
        return sum([
                stats[metric] * scale for metric, scale in self.config["fitness_config"].items()
            ])
    def compute_fitnesses(self):
        evaluated_population = self.evaluated_population()
        for individual_id, individual in evaluated_population.items():
            other_ids = list(evaluated_population.keys())
            other_ids.remove(individual_id)
            min_distance = 0 if self.config["novelty_n_samples"] == 0 else min([
                individual.pairwise_novelty(self.population[other_id])
                for other_id in np.random.choice(other_ids,
                                                 np.minimum(self.config["novelty_n_samples"],
                                                            len(evaluated_population) - 1),
                                                 replace=False)
            ])
            individual.evaluation_dict["novelty"] = min_distance
            individual.evaluation_dict["length"] = individual.action_sequence.seq_len

            individual.evaluation_dict["GA/TRUE_FITNESS"] = self.compute_true_fitness(individual.evaluation_dict)
            individual.evaluation_dict["GA/FITNESS"] = (
                    individual.evaluation_dict["GA/TRUE_FITNESS"]
                    + self.config["fitness_novelty_weight"] * individual.evaluation_dict["novelty"])

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

    def make_offspring(self, archive: "GoExploreArchive"):
        new_id = str(uuid.uuid4())[:8]


        self.population[new_id].evolve(new_id, parents=self.get_matting(), archive=archive)



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

        self.entries_hist = []

    def __getitem__(self, individual: Individual):
        if len(self.population) == self.max_entries:
            popped = self.entries_hist.pop(0)
            print("Popped", popped, "out of archive.")
            self.population.pop(popped)

        assert individual.ID is not None, "Non-initialized individual added to archive !"
        name = self.add_entry(individual)

        return self.population[name]

    def add_entry(self, individual: Individual):
        self.population[individual.ID].set_as(individual)
        self.entries_hist.append(individual.ID)
        return individual.ID

    def __len__(self):
        return len(self.population)


class GoExploreArchive(Archive):
    def __init__(self,
                 environment: gymnasium.Env,
                 config: dict,
                 max_size=4096,
                 base_starting_point=None
                 ):
        super().__init__(environment, config, max_size)
        self.relevant_features = config["go_explore_relevant_features"]
        self.base_starting_point=base_starting_point

        self.population = DefaultOrderedDict(lambda :
                                      {"cost": np.inf,
                                       "value": -np.inf,
                                       "action_sequence": None,
                                       "parent": None})


        self.stat_weights = {
            GoExplorePokemon.TIMES_CHOSEN                 : 1.,
            GoExplorePokemon.TIMES_SEEN                   : 15.,
        }

        self.state_stats = defaultdict(lambda: {
            GoExplorePokemon.TIMES_CHOSEN                 : 0.,
            GoExplorePokemon.TIMES_SEEN                   : 0,
        })

    def add_entry(self, individual: Individual):

        # Update state counts
        state_stats = individual.evaluation_dict["GO_EXPLORE/"+GoExplorePokemon.TIMES_SEEN]
        for state, count in state_stats.items():
            self.state_stats[state][GoExplorePokemon.TIMES_SEEN] += count

        prev_identifier = None

        for identifier, d in individual.evaluation_dict["GO_EXPLORE/key_states"].items():

            cost = d["cost"]
            stats = d["stats"]
            value = self.compute_true_fitness(stats)


            if identifier in self.population:
                elite = self.population[identifier]
                elite_value = elite["value"]
                elite_cost = elite["cost"]

                print(f"Better elite for {identifier} ?")
                print(cost, elite_cost)
                print(value, elite_value)

                # if ((value >= elite_value and cost < elite_cost)
                # or (value > elite_value and cost <= elite_cost)):
                if cost < elite_cost:

                    self.population[identifier]["value"] = value
                    self.population[identifier]["cost"] = cost
                    self.population[identifier]["action_sequence"] = individual.action_sequence.sequence[:cost]
                    self.population[identifier]["start_point"] = d["game_state"]
                    self.population[identifier]["parent"] = prev_identifier

                    def update_children(node):
                        for next_node in self.population:
                            if self.population[next_node]["parent"] == node:
                                self.population[next_node]["action_sequence"][:self.population[node]["cost"]] = self.population[node]["action_sequence"]
                                update_children(next_node)

                    update_children(identifier)

                    self.entries_hist.remove(identifier)
                    self.entries_hist.append(identifier)
                    print("Yes.")

                else:
                    print("No.")

            else:
                self.population[identifier]["value"] = value
                self.population[identifier]["cost"] = cost
                self.population[identifier]["action_sequence"] = individual.action_sequence.sequence[:cost]
                self.population[identifier]["start_point"] = d["game_state"]
                self.population[identifier]["parent"] = prev_identifier

                self.entries_hist.append(identifier)

            prev_identifier = identifier

        return identifier

    def compute_scores(self):
        scores = []
        for identifier, individual in self.population.items():

            state_stats = self.state_stats[identifier]

            score = 1e-5

            for stat_name, value in state_stats.items():
                score += self.stat_weights[stat_name] * np.sqrt(1 / (value + 1e-1))

            scores.append(score)

        return scores

    def get_probs(self):
        scores = self.compute_scores()

        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        return probs

    def sample_starting_point(self):

        if len(self.population) == 0:
            return self.base_starting_point

        probs = self.get_probs()

        starting_point_idx = np.random.choice(len(self.population), p=probs)
        starting_point_id = list(self.population.keys())[starting_point_idx]

        self.state_stats[starting_point_id][GoExplorePokemon.TIMES_CHOSEN] += 1

        return self.population[starting_point_id]


    def __repr__(self):

        string = "------------ARCHIVE------------\n\n"

        for (identifier, elite), p in zip(self.population.items(), self.get_probs()):
            string += (f"{identifier}->\t VALUE:{elite['value']:.3f},\t COST:{elite['cost']},\t SAMPLE CHANCE:{p:.3f}"
                       f"\n")
        return string


class GA:
    def __init__(self, env_cls, config):
        base_env = env_cls(config["env_config"])
        self.env_cls = env_cls
        self.population = Population(base_env, config)
        self.eval_workers = {w_id: Worker.as_remote().remote(w_id, env_cls, config) for w_id in range(config["num_workers"])}
        #self.eval_workers = mp.Pool(config["num_workers"], maxtasksperchild=1)
        self.available_worker_ids = {w_id for w_id in range(config["num_workers"])}

        self.go_explore_archive = GoExploreArchive(base_env, config, config["archive_size"], base_starting_point=base_env.base_starting_point)

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

                # res = []
                # for job in jobs:
                #     try:
                #         if job.ready():
                #             r = job.get()
                #             res.append(r)
                #             jobs.remove(job)
                #             done_jobs.append(jobs)
                #             print(r)
                #     except mp.TimeoutError as e:
                #         pass


                latest_done_jobs, jobs = ray.wait(
                    jobs,
                    num_returns=1,
                    timeout=None,
                    fetch_local=False
                )
                done_jobs.extend(latest_done_jobs)

                # print("done jobs:", len(done_jobs))
                done_workers = []

                for w_id, eval_dict in ray.get(latest_done_jobs):
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
            # def eval_individual():
            #     self.population[next_individual_id].eval(self.env_cls)

            #jobs.append(self.eval_workers.apply_async(self.population[next_individual_id].eval, (self.env_cls, w_id)))

            jobs.append(
                self.eval_workers[w_id].eval.remote(
                    self.population[next_individual_id]
                ))

            if len(jobs) == max_jobs:
                break

        return jobs

    def __call__(self):
        past_individuals = list(self.population.population.keys())

        while True:

            for w_id in self.available_worker_ids:
                new_id = self.population.make_offspring(self.go_explore_archive)
                self.to_eval_queue.append(new_id)

            if len(self.population.evaluated_population()) >= 2 * self.population.size:
                self.population.select()

                evaluated = self.population.evaluated_population()
                #print("Best individual so far:")
                best = sorted(evaluated, key=lambda i_id: -evaluated[i_id].evaluation_dict["GA/FITNESS"])[0]
                self.population.save_individual(best, "best_individual.pkl")

                for ID, individual in self.population.evaluated_population().items():
                    if ID not in past_individuals:
                        self.go_explore_archive[individual]
                past_individuals = list(self.population.population.keys())

                print(self.go_explore_archive)
                lengths = [individual.action_sequence.seq_len for individual in self.population.population.values()]
                print()
                print("Seq length stats:", np.min(lengths), np.mean(lengths), np.max(lengths))


            for evaluated_individual in next(self.runner):
                self.population[evaluated_individual.ID].evaluation_dict = evaluated_individual.evaluation_dict


    def init_eval(self):
        self.to_eval_queue = list(self.population.population.keys())
        self.num_expected_evals = len(self.population)

        for evaluated_individual in next(self.runner):
            self.population[evaluated_individual.ID].evaluation_dict = evaluated_individual.evaluation_dict

        self.num_expected_evals = 1

        self.population.compute_fitnesses()
        for ID, individual in self.population:
            self.go_explore_archive[individual]

        print(self.go_explore_archive)



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
        "action_sequence_limits"   : (128, 2048*8),
        "env_config"               : {
            "init_state"  : "deepred_post_parcel_pokeballs.state",
            "session_path": Path("sessions/tests"),
            "gb_path"     : "pokered.gbc",
            "render"      : False
        },
        "population_size"          : 500,
        "num_workers"              : os.cpu_count()-1,
        "fitness_config"           : {
            "episode_reward": 5.,
            BADGE_SUM       : 100.,

            CAUGHT_POKEMONS : 1.0,
            SEEN_POKEMONS   : 1.0,
            MAPS_VISITED    : 0.1,

        },
        "fitness_novelty_weight": 5e-4,
        "novelty_n_samples"        : 0,
        "crossover_n_points"       : 1,
        "mutation_rate"            : 0.05,
        "subsequence_mutation_rate": 1e-3,
        "max_subsequence_length"   : 64,

        "go_explore_relevant_features": (
            MAP_ID,
            TOTAL_EVENTS_TRIGGERED,
            BADGE_SUM
        ),
        "archive_size": 10_000,
        "base_starting_point_sample_chance": 1.,
    }

    ray.init()
    runner = GA(env_cls=PkmnRedEnvNoRender, config=config)
    runner.initialize_population()
    runner.init_eval()
    print("init done")
    runner()
