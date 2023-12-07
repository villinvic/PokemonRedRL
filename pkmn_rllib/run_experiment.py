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


run_steps = 2048*8

sess_path = f'sessions/session_{str(uuid.uuid4())[:8]}'

args = get_args('run_baseline.py', ep_length=run_steps, sess_path=sess_path)

env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': 'has_pokedex_nballs', 'max_steps': run_steps,
                'print_rewards': False, 'save_video': True, 'fast_video': True,
                'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 70_000_000.,
                'knn_elements': 20_000,
                'additional_steps_per_episode': 1
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

num_workers = 124
num_envs_per_worker = 1
rollout_fragment_length = 1024

config = VmpoConfig().training(
    eps_eta=2e-2,
    eps_alpha=2e-3,
    alpha=5.,
    target_network_update_freq=1000, #1536,
    replay_proportion=0.5,
    entropy_coeff=1e-4,
    learner_queue_size=64,
    lr=8e-4,
    statistics_lr=3e-1,
    momentum=0.,
    epsilon=1e-5,
    decay=0.99,
    grad_clip=1.,
    opt_type="rmsprop",
    train_batch_size=8096//2,
    #num_sgd_iter=1,
    #minibatch_buffer_size=128,
    gamma=0.998,
    model={
        "custom_model": "pokemon_lstm_model",
        "conv_filters": [
            [32, [8, 8], 4, "valid"],
            [64, [4, 4], 2, "valid"],
            [64, [3, 3], 1, "valid"],
        ],
        "fcnet_size": 512,
        "lstm_size": 512,
        #"flag_embedding_size": 64,
        "max_seq_lens": 64,
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
).reporting(min_train_timesteps_per_iteration=20, metrics_episode_collection_timeout_s=4*60
).experimental(_disable_preprocessor_api=True,
).resources(num_gpus=1
).framework(framework="tf")



ckpt_config = air.CheckpointConfig(
    num_to_keep=3,
    checkpoint_frequency=50,
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

# tuner = tune.Tuner(
#         trainable=Vmpo,
#         param_space=config.to_dict(),
#         run_config=run_config,
# )
# tuner.fit()

exp = tune.run(
        Vmpo,
        name=run_config.name,
        config=config.to_dict(),
        checkpoint_at_end=ckpt_config.checkpoint_at_end,
        checkpoint_freq=ckpt_config.checkpoint_frequency,
        keep_checkpoints_num=ckpt_config.checkpoint_at_end,
        stop=stopping_config,
        local_dir="rllib_runs",
        #restore="/home/goji/Documents/PokemonRedRL/rllib_runs/v1_2/Vmpo_PokemonRed_d7cc3_00000_0_2023-12-05_12-45-52/checkpoint_000400"
        #resume=True
    )


