#!/usr/bin/env python3
from datetime import datetime

import gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import cyberbattle
from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from cyberbattle.agents.sb3.ppo_cat.policies import MlpPolicy
from cyberbattle.agents.sb3.ppo_cat.ppo_cat import CATPPO
from env_wrapper import SB3EnvWrapper, SB3MultiDiscreteActionModel, mask_fn, mask_fn_2


def main():
    env = gym.make('CyberBattleChain-v0',
                   size=12,
                   attacker_goal=cyberbattle.AttackerGoal(
                       own_atleast_percent=1.0),
                   maximum_node_count=16,
                   maximum_total_credentials=25,
                   maximum_steps=1000,
                   )

    ep = EnvironmentBounds.of_identifiers(
        maximum_total_credentials=25,
        maximum_node_count=16,
        identifiers=env.identifiers
    )
    env = SB3EnvWrapper(env, ep)
    # env = Monitor(env, filename='ppo.log')
    # env = ActionMasker(env, mask_fn_2)

    # model = PPO('MlpPolicy', env, verbose=2, tensorboard_log='./ppo_tboard/')
    # model = MaskablePPO('MlpPolicy', env, verbose=2, tensorboard_log='./ppo_mask_tboard/')
    model = CATPPO(MlpPolicy, env, verbose=2, tensorboard_log='./ppo_mask_tboard/')
    model.learn(
        total_timesteps=500_000,
        # eval_freq=15_000,
        # n_eval_episodes=5,
        # eval_log_path='./logs/',
        # eval_env=eval_env,
    )
    now = datetime.now()
    date_str = now.strftime("%d-%-m-%y.%H-%M")
    model.save(f'ppo_cyberbattle-{date_str}.zip')

    # model = PPO.load("ppo_cyberbattle-03-5-22.08-54.zip")
    model = MaskablePPO.load("ppo_cyberbattle-09-5-22.22-44.zip")
    obs = env.reset()
    done = False

    good_actions = []
    step = 1
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        gym_action = env.action_space.get_gym_action(action)

        if reward > 0.0:
            good_actions.append((gym_action, reward))
        print(f'step: {step}, action: {gym_action}, reward: {reward}')
        step += 1

    print(good_actions)

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


if __name__ == "__main__":
    main()
