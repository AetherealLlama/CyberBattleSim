#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from typing import Tuple

import gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymEnv

import cyberbattle
from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from cyberbattle._env.defender import DefenderAgent, ScanAndReimageCompromisedMachines
from cyberbattle.agents.sb3.env_wrapper import SB3EnvWrapper
from cyberbattle.agents.sb3.ppo_cat.ppo_cat import CATPPO
from cyberbattle.agents.sb3.ppo_cat.utils import get_vat
from rich import print

ENV_ID = 'CyberBattleChain-v0'
ENV_SIZE = 8
ATTACKER_GOAL = cyberbattle.AttackerGoal(own_atleast_percent=1.0)
MAXIMUM_NODE_COUNT = 11
MAXIMUM_TOTAL_CREDENTIALS = 15
MAX_STEPS = 2500
REWARD_MULTIPLIER = 50.0

NUM_EVAL_EPISODES = 250

DEFENDER_INSTALLED = False
DEFENDER_DETECT_PROB = 0.2
DEFENDER_SCAN_CAPACITY = 1
DEFENDER_SCAN_FREQUENCY = 75


def make_env(env_size: int = ENV_SIZE) -> Tuple[GymEnv, GymEnv]:
    defender_agent: DefenderAgent = ScanAndReimageCompromisedMachines(probability=DEFENDER_DETECT_PROB,
                                                                      scan_capacity=DEFENDER_SCAN_CAPACITY,
                                                                      scan_frequency=DEFENDER_SCAN_FREQUENCY)
    env = gym.make(
        ENV_ID,
        size=env_size,
        attacker_goal=ATTACKER_GOAL,
        maximum_node_count=MAXIMUM_NODE_COUNT,
        maximum_total_credentials=MAXIMUM_TOTAL_CREDENTIALS,
        maximum_steps=MAX_STEPS,
        reward_multiplier=REWARD_MULTIPLIER,
        defender_agent=defender_agent if DEFENDER_INSTALLED else None,
    )
    ep = EnvironmentBounds.of_identifiers(
        maximum_node_count=MAXIMUM_NODE_COUNT,
        maximum_total_credentials=MAXIMUM_TOTAL_CREDENTIALS,
        identifiers=env.identifiers,
    )
    wenv = SB3EnvWrapper(env, ep)
    return wenv, env


def run_random_episode(env: GymEnv, valid_actions: bool = True) -> Tuple[float, int]:
    obs = env.reset()
    done = False

    total_reward = 0.0
    step = 0

    while not done:
        if valid_actions:
            action = env.sample_valid_action()
        else:
            action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        total_reward += reward
        step += 1

    return total_reward, step


def run_model_episode(model: CATPPO, env: GymEnv) -> Tuple[float, int]:
    obs = env.reset()
    done = False

    total_reward = 0.0
    step = 0

    while not done:
        valid_action_tree = np.array([get_vat(env)])
        action, _ = model.predict(obs, valid_action_trees=valid_action_tree)
        obs, reward, done, info = env.step(action)

        step += 1
        total_reward += reward

    return total_reward, step


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = sys.argv[1]

    for env_size in range(0, 8, 2):
        wenv, env = make_env(env_size)
        wenv = Monitor(wenv, "ppo_eval")

        model = CATPPO.load(model_name, env=wenv, device=device, print_system_info=True)

        print(f"env_size {env_size} - Evaluating the model...")
        model_rewards, model_lengths = [], []
        for i in range(NUM_EVAL_EPISODES):
            reward, length = run_model_episode(model, wenv)
            model_rewards.append(reward)
            model_lengths.append(length)

            print(f"s {env_size}, model #{i}/{NUM_EVAL_EPISODES} -- r: {reward}, l: {length}")

        results_model = {
            "rewards": model_rewards,
            "lengths": model_lengths,
        }
        with open(f"ppo_results_model_s{env_size}_1.json", 'w') as out:
            json.dump(results_model, out, indent=4)

        print("Evaluating random agent with valid actions...")
        random_valid_rewards, random_valid_lengths = [], []
        for i in range(NUM_EVAL_EPISODES):
            reward, length = run_random_episode(env, True)
            random_valid_rewards.append(reward)
            random_valid_lengths.append(length)
            print(f"s {env_size}, random v #{i}/{NUM_EVAL_EPISODES} -- r: {reward}, l: {length}")

        results_random_valid = {
            "rewards": random_valid_rewards,
            "lengths": model_lengths,
        }
        with open(f"ppo_results_random_v_s{env_size}_1.json", 'w') as out:
            json.dump(results_random_valid, out, indent=4)

        print("Evaluating random agent...")
        random_rewards, random_lengths = [], []
        for i in range(NUM_EVAL_EPISODES):
            reward, length = run_random_episode(env, False)
            random_rewards.append(reward)
            random_lengths.append(length)
            print(f"s {env_size}, random #{i}/{NUM_EVAL_EPISODES} -- r: {reward}, l: {length}")

        results_random = {
            "rewards": random_rewards,
            "lengths": random_lengths,
        }
        with open(f"ppo_results_random_s{env_size}_1.json", 'w') as out:
            json.dump(results_random, out, indent=4)

        results = {
            "model": {
                "rewards": model_rewards,
                "lengths": model_lengths,
            },
            "random_valid": {
                "rewards": random_valid_rewards,
                "lengths": random_valid_lengths,
            },
            "random": {
                "rewards": random_rewards,
                "lengths": random_lengths,
            },
        }
        now = datetime.now()
        date_str = now.strftime("%d-%-m-%y.%H-%M")
        filename = f"ppo_results_s{env_size}_{ENV_ID}-{date_str}.json"
        with open(filename, 'w') as out:
            json.dump(results, out, indent=4)


if __name__ == '__main__':
    main()
