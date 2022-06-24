#!/usr/bin/env python3
from datetime import datetime
from typing import Optional, Callable

import gym
import torch
import torch.nn as nn
import wandb
from rich import print
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymEnv
from wandb.integration.sb3 import WandbCallback

import cyberbattle
from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from cyberbattle._env.defender import DefenderAgent, ScanAndReimageCompromisedMachines
from cyberbattle.agents.sb3.env_wrapper import SB3EnvWrapper
from cyberbattle.agents.sb3.ppo_cat.callbacks import ProgressBarManager
from cyberbattle.agents.sb3.ppo_cat.evaluation import evaluate_policy
from cyberbattle.agents.sb3.ppo_cat.policies import MlpPolicy
from cyberbattle.agents.sb3.ppo_cat.ppo_cat import CATPPO

ENV_ID = 'CyberBattleSB3-v0'
ENV_SIZE = 8
ATTACKER_GOAL = cyberbattle.AttackerGoal(own_atleast_percent=1.0)
MAXIMUM_NODE_COUNT = 11
MAXIMUM_TOTAL_CREDENTIALS = 25
MAX_STEPS = 10
REWARD_MULTIPLIER = 50.0

NUM_ENVS = 5
NUM_TIMESTEPS = 150_000
NUM_EVALUATIONS = 6
EVAL_FREQ = int(NUM_TIMESTEPS / NUM_EVALUATIONS)
NUM_EVAL_EPISODES = 50

DEFENDER_INSTALLED = True
DEFENDER_DETECT_PROB = 0.2
DEFENDER_SCAN_CAPACITY = 1
DEFENDER_SCAN_FREQUENCY = 100


def make_env(rank: int, seed: Optional[int] = None, eval_env: bool = False) -> Callable[[], GymEnv]:
    def _init() -> gym.Env:
        defender_agent: DefenderAgent = ScanAndReimageCompromisedMachines(probability=DEFENDER_DETECT_PROB, scan_capacity=DEFENDER_SCAN_CAPACITY, scan_frequency=DEFENDER_SCAN_FREQUENCY)
        env = gym.make(
            ENV_ID,
            # size=ENV_SIZE,
            size=10,
            prob=0.15,
            attacker_goal=ATTACKER_GOAL,
            maximum_node_count=MAXIMUM_NODE_COUNT,
            maximum_total_credentials=MAXIMUM_TOTAL_CREDENTIALS,
            maximum_steps=MAX_STEPS,
            reward_multiplier=REWARD_MULTIPLIER,
            defender_agent=defender_agent if DEFENDER_INSTALLED else None,
        )
        if seed is not None:
            env.seed(seed + rank)
        ep = EnvironmentBounds.of_identifiers(
            maximum_node_count=MAXIMUM_NODE_COUNT,
            maximum_total_credentials=MAXIMUM_TOTAL_CREDENTIALS,
            identifiers=env.identifiers,
        )
        env = SB3EnvWrapper(env, ep)
        log_name = f'ppo_cat_eval_{rank}' if eval_env else f'ppo_cat_{rank}'
        env = Monitor(env, log_name)
        return env

    return _init


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    # eval_env = SubprocVecEnv([make_env(0, eval_env=True)])
    env = make_env(0)()
    eval_env = make_env(0, eval_env=True)()

    activation_fns = {"tanh": nn.Tanh, "relu": nn.ReLU}

    config = {
        'policy_type': 'MlpPolicy',
        'env_name': ENV_ID,
        'total_timesteps': NUM_TIMESTEPS,
        # 'env_size': ENV_SIZE,
        'max_node_count': MAXIMUM_NODE_COUNT,
        'max_total_credentials': MAXIMUM_TOTAL_CREDENTIALS,
        'reward_multiplier': REWARD_MULTIPLIER,
        'num_evaluations': NUM_EVALUATIONS,
        'num_eval_episodes': NUM_EVAL_EPISODES,
        'defender_installed': DEFENDER_INSTALLED,
        'defender_detect_probability': DEFENDER_DETECT_PROB,
        'defender_scan_capacity': DEFENDER_SCAN_CAPACITY,
        'defender_scan_frequency': DEFENDER_SCAN_FREQUENCY,

        'learning_rate': 0.0001,
        'gamma': 0.93,
        'n_steps': 16,
        'batch_size': 2,
        'n_epochs': 35,
        'net_arch': [dict(pi=[1024, 512, 256], vf=[1024, 512, 256])],
        'activation_fn': 'tanh'
    }

    # wandb.init(
    #     project='sb3',
    #     config=config,
    #     sync_tensorboard=True,
    #     save_code=True,
    # )

    model = CATPPO(
        MlpPolicy,
        env,
        verbose=2,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        policy_kwargs={
            "net_arch": config['net_arch'],
            "activation_fn": activation_fns[config['activation_fn']]
        },
        tensorboard_log='./ppo_mask_tboard/',
        device=device,
    )
    with ProgressBarManager(NUM_TIMESTEPS) as c:
        # wandb_callback = WandbCallback(verbose=2)
        model.learn(
            total_timesteps=NUM_TIMESTEPS,
            callback=[
                # wandb_callback,
                c,
            ],
            eval_freq=EVAL_FREQ,
            n_eval_episodes=NUM_EVAL_EPISODES,
            eval_log_path='./logs/',
            eval_env=eval_env,
        )
    now = datetime.now()
    date_str = now.strftime("%d-%-m-%y.%H-%M")
    model_file_name = f'ppo_cat_cyberbattle{"_def" if DEFENDER_INSTALLED else ""}-{date_str}.zip'
    model.save(model_file_name)
    print(f'saved model as `{model_file_name}`')

    mean_reward, std_reward, mean_length = evaluate_policy(model, eval_env, n_eval_episodes=NUM_EVAL_EPISODES, return_mean_ep_length=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    print(f"mean_episode_length={mean_length}")


if __name__ == "__main__":
    main()
