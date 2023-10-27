import uuid
from baselines.argparse_pokemon import get_args, change_env

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray import air, tune

from pkmn_env.red import PkmnRedEnv
from pkmn_rllib.rllib.models.PokemonBaseModel import PokemonBaseModel
from pkmn_rllib.rllib.models.PokemonLstmModel import PokemonLstmModel
from pkmn_rllib.rllib.vmpo.Vmpo import VmpoConfig, Vmpo
from pkmn_rllib.rllib.vmpo.rllib_callbacks import PokemonCallbacks

run_steps = 512 * 4 #16384

sess_path = f'session_{str(uuid.uuid4())[:8]}'

args = get_args('run_baseline.py', ep_length=run_steps, sess_path=sess_path)

env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': 'has_pokedex_nballs.state', 'max_steps': run_steps,
                'print_rewards': False, 'save_video': False, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 10_000_000.,
                'knn_elements': 1000,
            }

env_config = change_env(env_config, args)

def make_env(env_config_):
    return PkmnRedEnv(config=env_config_)

register_env("PokemonRed", make_env)

ModelCatalog.register_custom_model(
        "pokemon_base_model",
        PokemonBaseModel,
    )

ModelCatalog.register_custom_model(
        "pokemon_lstm_model",
        PokemonLstmModel,
    )

num_workers = 120
rollout_fragment_length = 128

config = VmpoConfig().training(
    eps_eta=2e-2,
    eps_alpha=1e-2,
    alpha=5.,
    target_network_update_freq=10,
    replay_proportion=0.,
    entropy_coeff=0.,
    learner_queue_size=128,
    lr=8e-4,
    statistics_lr=1e-1,
    momentum=0.,
    epsilon=1e-5,
    decay=0.99,
    grad_clip=10.,
    opt_type="rmsprop",
    train_batch_size=num_workers*rollout_fragment_length,
    gamma=0.993,
    model={
        "custom_model": "pokemon_lstm_model",
        "conv_filters": [
            [16, [8, 8], 4],
            [32, [4, 4], 2],
        ],
        "fcnet_size": 64,
        "lstm_size": 128,
    }
).rollouts(
    num_rollout_workers=num_workers,
    sample_async=True,
    create_env_on_local_worker=False,
    rollout_fragment_length=rollout_fragment_length,
    batch_mode="truncate_episodes"
).callbacks(PokemonCallbacks
).environment(
    env="PokemonRed",
    env_config=env_config
).reporting(min_sample_timesteps_per_iteration=20
).experimental(_disable_preprocessor_api=True,
).resources(num_gpus=1
).framework(framework="tf")



ckpt_config = air.CheckpointConfig(
    num_to_keep=3,
    checkpoint_frequency=1000,
    checkpoint_at_end=True)

stopping_config = {
            "timesteps_total": 1_000_000_000,
        }

run_config = air.RunConfig(
    name="v1_2",
    local_dir="rllib_runs",
    stop=stopping_config,
    checkpoint_config=ckpt_config,
    verbose=0
)

tuner = tune.Tuner(
        trainable=Vmpo,
        param_space=config.to_dict(),
        run_config=run_config,
)

tuner.fit()


