#!/usr/bin/env python3
from typing import Dict, Any, Optional, Callable

import gym
import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

import cyberbattle
from cyberbattle._env.cyberbattle_env import EnvironmentBounds
from cyberbattle.agents.sb3.env_wrapper import SB3EnvWrapper
from cyberbattle.agents.sb3.ppo_cat.callbacks import VATEvalCallback
from cyberbattle.agents.sb3.ppo_cat.policies import MlpPolicy
from cyberbattle.agents.sb3.ppo_cat.ppo_cat import CATPPO

N_TRIALS = 20
N_STARTUP_TRIALS = 3
N_EVALUATIONS = 2
N_TIMESTEPS = int(5e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

ENV_ID = 'CyberBattleChain-v0'

DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    'policy': MlpPolicy,
    'verbose': 2,
}

NUM_ENVS = 5
ENV_SIZE = 10
MAX_STEPS = 2000
REWARD_MULTIPLIER = 10.0
ATTACKER_GOAL = cyberbattle.AttackerGoal(own_atleast_percent=1.0)
MAXIMUM_NODE_COUNT = 16
MAXIMUM_TOTAL_CREDENTIALS = 25


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    learning_rate = trial.suggest_float("lr", 01e-5, 1, log=True)
    n_steps = 2 ** trial.suggest_int("n_steps", 8, 12)
    batch_size = 2 ** trial.suggest_int("batch_size", 7, 12)
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])

    trial.set_user_attr("n_steps_", n_steps)
    trial.set_user_attr("batch_size_", batch_size)

    net_arch = [dict(pi=[1024, 512, 128], vf=[1024, 512, 128])]
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
        "tensorboard_log": './ppo_mask_tboard/',
    }


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


class TrialEvalCallback(VATEvalCallback):
    def __init__(
        self,
        eval_env: GymEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            use_vat=True,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(sample_ppo_params(trial))

    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    kwargs['env'] = env

    model = CATPPO(**kwargs)
    eval_env = make_env(66)()

    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=False
    )
    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        print(e)
        nan_encountered = True
    finally:
        model.env.close()
        eval_env.close()

    if nan_encountered:
        return float('nan')

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    return eval_callback.last_mean_reward


def main():
    torch.set_num_threads(1)
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize')
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print(f"Number of finished trials: {len(study.trials)}")

    print("----- Best trial -----")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")

    print("  User attrs:")
    for k, v in trial.user_attrs.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
