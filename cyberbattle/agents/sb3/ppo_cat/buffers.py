from typing import NamedTuple, Optional, Generator, List, Dict, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class VATRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    valid_action_trees: th.Tensor


class VATRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the valid action trees associated with each observation
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(self, *args, **kwargs):
        self.valid_action_trees = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        if isinstance(self.action_space, spaces.MultiDiscrete):
            vat_dims = len(self.action_space.nvec)
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")

        self.vat_dims = vat_dims
        self.valid_action_trees = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.object)

        super().reset()

    def add(self, *args, valid_action_trees: np.ndarray = None, **kwargs) -> None:
        if valid_action_trees is not None:
            self.valid_action_trees[self.pos] = valid_action_trees

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[VATRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "valid_action_trees"
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> VATRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return VATRolloutBufferSamples(*(*map(self.to_torch, data), self.valid_action_trees[batch_inds]))
