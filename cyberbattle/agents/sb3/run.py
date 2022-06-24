#!/usr/bin/env python3
import sys
from typing import Optional, Callable

import gym
import numpy as np
import torch
from rich import print
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymEnv

import cyberbattle
from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from cyberbattle.agents.sb3.env_wrapper import SB3EnvWrapper
from cyberbattle.agents.sb3.ppo_cat.evaluation import evaluate_policy
from cyberbattle.agents.sb3.ppo_cat.ppo_cat import CATPPO
from cyberbattle.agents.sb3.ppo_cat.utils import get_vat
from cyberbattle._env.defender import DefenderAgent, ScanAndReimageCompromisedMachines

ENV_ID = 'CyberBattleChain-v0'
ENV_SIZE = 10
ATTACKER_GOAL = cyberbattle.AttackerGoal(own_atleast_percent=1.0)
MAXIMUM_NODE_COUNT = 20
MAXIMUM_TOTAL_CREDENTIALS = 25
MAX_STEPS = 1000
REWARD_MULTIPLIER = 100.0

NUM_EVAL_EPISODES = 1000

DEFENDER_INSTALLED = False
DEFENDER_DETECT_PROB = 0.2
DEFENDER_SCAN_CAPACITY = 1
DEFENDER_SCAN_FREQUENCY = 75


def make_env(rank: int, seed: Optional[int] = None) -> Callable[[], GymEnv]:
    def _init() -> gym.Env:
        defender_agent: DefenderAgent = ScanAndReimageCompromisedMachines(probability=0.2, scan_capacity=1, scan_frequency=75)
        env = gym.make(
            ENV_ID,
            size=ENV_SIZE,
            attacker_goal=ATTACKER_GOAL,
            maximum_node_count=MAXIMUM_NODE_COUNT,
            maximum_total_credentials=MAXIMUM_TOTAL_CREDENTIALS,
            maximum_steps=MAX_STEPS,
            reward_multiplier=REWARD_MULTIPLIER,
            # defender_agent=defender_agent,
        )
        if seed is not None:
            env.seed(seed + rank)
        ep = EnvironmentBounds.of_identifiers(
            maximum_node_count=MAXIMUM_NODE_COUNT,
            maximum_total_credentials=MAXIMUM_TOTAL_CREDENTIALS,
            identifiers=env.identifiers,
        )
        env = SB3EnvWrapper(env, ep)
        return env

    return _init


def run_episode(model: CATPPO, env: GymEnv):
    obs = env.reset()
    done = False

    good_actions = []
    step = 1
    total_reward = 0
    while not done:
        vat = np.array([get_vat(env)])
        action, _states = model.predict(obs, valid_action_trees=vat)
        obs, reward, done, info = env.step(action)
        gym_action = env.action_space.get_gym_action(action)

        if reward > 0.0:
            good_actions.append((gym_action, reward))
        print(f'step: {step}, action: {gym_action}, reward: {reward}')
        step += 1
        total_reward += reward

    print(good_actions)
    print(f'total reward: {total_reward}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = sys.argv[1]

    env = Monitor(make_env(0)(), "ppo_cat_eval")
    model = CATPPO.load(model_name, env=env, device=device, print_system_info=True)

    run_episode(model, env)
    print('sup mcnuggz')

    episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=NUM_EVAL_EPISODES,
                                                           return_episode_rewards=True)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    print(f"mean_episode_length={mean_length} +/- {std_length}")

    print(f"episode_rewards: {episode_rewards}")
    print(f"episode_lengths: {episode_lengths}")


if __name__ == '__main__':
    main()
