from setuptools import setup

setup(
    name="pkmn_rllib",
    version="0.0.1",
    python_requires='>=3.10',
    install_requires=[
        'absl-py',
        'chex<0.1.81',  # Incompatible with tensorflow 2.13 (due to numpy req).
        'dm-env',
        'dmlab2d',
        'dm-tree',
        'immutabledict',
        'ml-collections',
        'networkx',
        'numpy==1.24.3',
        'opencv-python',
        'pandas',
        'pygame',
        'reactivex',
        'tensorflow==2.11.1',
        'tensorflow-probability',
        'torch==2.0.1',
        'ray[all]==2.6.1',
        'gymnasium',
        'matplotlib',
        'pydantic==1.10.12',
        'wandb',
        "pyboy==1.5.6",
        "einops==0.6.1",
        "hnswlib==0.7.0",
        "mediapy @ git+https://github.com/PWhiddy/mediapy.git@45101800d4f6adeffe814cad93de1db67c1bd614"
        ""
    ],

    packages=["pkmn_rllib"]
)