#!/usr/bin/env python3
from datetime import datetime
from typing import Optional, Callable

import gym
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import cyberbattle
from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from cyberbattle.agents.sb3.ppo_cat.policies import MlpPolicy
from cyberbattle.agents.sb3.ppo_cat.ppo_cat import CATPPO
from env_wrapper import SB3EnvWrapper

ENV_ID = 'CyberBattleChain-v0'
ENV_SIZE = 12
ATTACKER_GOAL = cyberbattle.AttackerGoal(own_atleast_percent=1.0)
MAXIMUM_NODE_COUNT = 20
MAXIMUM_TOTAL_CREDENTIALS = 25
MAX_STEPS = 2000
REWARD_MULTIPLIER = 10.0

NUM_ENVS = 5
NUM_TIMESTEPS = 100_000
NUM_EVALUATIONS = 2
EVAL_FREQ = int(NUM_TIMESTEPS / NUM_EVALUATIONS)
NUM_EVAL_EPISODES = 1


def make_env(rank: int, seed: Optional[int] = None) -> Callable[[], GymEnv]:
    def _init() -> gym.Env:
        env = gym.make(
            ENV_ID,
            size=ENV_SIZE,
            attacker_goal=ATTACKER_GOAL,
            maximum_node_count=MAXIMUM_NODE_COUNT,
            maximum_total_credentials=MAXIMUM_TOTAL_CREDENTIALS,
            maximum_steps=MAX_STEPS,
            reward_multiplier=REWARD_MULTIPLIER
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


def main():
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    eval_env = SubprocVecEnv([make_env(66)])

    # model = PPO('MlpPolicy', env, verbose=2, tensorboard_log='./ppo_tboard/')
    model = CATPPO(
        MlpPolicy,
        env,
        verbose=2,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=1024,
        n_epochs=25,
        policy_kwargs={
            "net_arch": [dict(pi=[1024, 512, 128], vf=[1024, 512, 128])]
        },
        tensorboard_log='./ppo_mask_tboard/'
    )
    model.learn(
        total_timesteps=NUM_TIMESTEPS,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=NUM_EVAL_EPISODES,
        eval_log_path='./logs/',
        eval_env=eval_env,
    )
    now = datetime.now()
    date_str = now.strftime("%d-%-m-%y.%H-%M")
    model.save(f'ppo_cat_cyberbattle-{date_str}.zip')

    # model = PPO.load("ppo_cyberbattle-03-5-22.08-54.zip")
    # model = MaskablePPO.load("ppo_cyberbattle-09-5-22.22-44.zip")
    # model = CATPPO.load("ppo_cat_cyberbattle-19-5-22.08-20.zip")
    # obs = env.reset()
    # done = False
    #
    # good_actions = []
    # step = 1
    # total_reward = 0
    # while not done:
    #     vat = np.array([get_vat(env)])
    #     action, _states = model.predict(obs, valid_action_trees=vat)
    #     obs, reward, done, info = env.step(action)
    #     gym_action = env.action_space.get_gym_action(action)
    #
    #     if reward > 0.0:
    #         good_actions.append((gym_action, reward))
    #     print(f'step: {step}, action: {gym_action}, reward: {reward}')
    #     step += 1
    #     total_reward += reward
    #
    # print(good_actions)
    # print(f'total reward: {total_reward}')

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


if __name__ == "__main__":
    main()
