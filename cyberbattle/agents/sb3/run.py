#!/usr/bin/env python3
import gym
from stable_baselines3.ppo import PPO

import cyberbattle
from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from env_wrapper import SB3EnvWrapper


def main():
    env = gym.make('CyberBattleChain-v0',
                   size=6,
                   attacker_goal=cyberbattle.AttackerGoal(
                       own_atleast_percent=1.0),
                   maximum_node_count=8,
                   maximum_total_credentials=25)
    ep = EnvironmentBounds.of_identifiers(
        maximum_total_credentials=25,
        maximum_node_count=8,
        identifiers=env.identifiers
    )
    env = SB3EnvWrapper(env, ep)
    # env = ActionMasker(env, mask_fn)

    # model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10)

    # check_env(env)


if __name__ == "__main__":
    main()
