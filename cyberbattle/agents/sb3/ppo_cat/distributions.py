from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from sb3_contrib.common.maskable.distributions import MaskableCategorical
from stable_baselines3.common.distributions import Distribution


class VATDistribution(Distribution, ABC):
    @abstractmethod
    def apply_vat(self, vat: Optional[np.ndarray]) -> None:
        pass


class VATMultiCategoricalDistribution(VATDistribution):
    """
    MultiCategorical distribution for multi discrete actions. Supports invalid action masking.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super().__init__()
        self.distributions: List[MaskableCategorical] = []
        self.valid_action_trees: Optional[np.ndarray] = None
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "VATMultiCategoricalDistribution":
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, sum(self.action_dims))

        self.distributions = [
            MaskableCategorical(logits=split) for split in th.split(reshaped_logits, tuple(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        logp = th.zeros([self.valid_action_trees.shape[0]])
        entropy = th.zeros([self.valid_action_trees.shape[0]])
        split_dists = [VATMultiCategoricalDistribution._split_categorical(d) for d in self.distributions]
        # transpose it
        split_dists = list(map(list, zip(*split_dists)))
        for idx, (action, vat, dists) in enumerate(zip(actions, self.valid_action_trees, split_dists)):
            action_logp, action_entropy = VATMultiCategoricalDistribution._logp_and_entropy_for_action(action, vat,
                                                                                                       dists)
            logp[idx] = action_logp.sum()
            entropy[idx] = action_logp.sum()

        return logp, entropy

    @staticmethod
    def _logp_and_entropy_for_action(action: th.Tensor, vat: np.ndarray, dists: List[MaskableCategorical]):
        assert isinstance(vat[0], Dict)
        logp = th.zeros([len(action)])
        entropy = th.zeros([len(action)])

        subtree = vat[0]
        action_type, source_node, target_node, local_vuln, remote_vuln, port, cred = range(7)
        sel_action = action[action_type]
        options = list(subtree.keys())
        # mask = np.full_like(dists[action_type].logits, False, dtype=bool)
        mask = np.full(tuple(dists[action_type].logits.shape), False, dtype=bool)
        mask[options] = True
        dists[action_type].apply_masking(mask)

        sel_action_logp = dists[action_type].log_prob(sel_action)
        logp[action_type] = sel_action_logp
        entropy[action_type] = dists[action_type].entropy()

        subtree = subtree[int(sel_action)]
        other_parts = {
            0: [source_node, local_vuln],
            1: [source_node, target_node, remote_vuln],
            2: [source_node, port, cred],
        }
        for part in other_parts[int(sel_action)]:
            options = list(subtree.keys())
            selection = action[part]
            # mask = np.full_like(dists[part].logits, False, dtype=bool)
            mask = np.full(tuple(dists[part].logits.shape), False, dtype=bool)
            mask[options] = True
            dists[part].apply_masking(mask)

            lp = dists[part].log_prob(selection)
            logp[part] = lp
            entropy[part] = dists[part].entropy()

            subtree = subtree[int(selection)]

        return logp, entropy

    # def _logp_and_entropy_for_action_part(self, subtree: Dict, sampled: th.Tensor, dist: MaskableCategorical) -> Tuple[th.Tensor, th.Tensor]:
    #     options = list(subtree.keys())
    #     mask = np.full_like(dist.logits[0], False, dtype=bool)
    #     mask[options] = True
    #     self.distributions[dist_pos].apply_masking(mask)
    #
    #     logp = self.distributions[dist_pos].log_prob(sampled)
    #     entropy = self.distributions[dist_pos].entropy()
    #     return logp, entropy

    def entropy(self) -> th.Tensor:
        # need an action for that one
        raise NotImplementedError

    def sample(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return self._mask_and_sample(False)

    def mode(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return self._mask_and_sample(True)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob[0]

    def apply_vat(self, vat: Optional[np.ndarray]) -> None:
        self.valid_action_trees = vat

    def _mask_and_sample(self, deterministic=False) -> th.Tensor:
        assert self.valid_action_trees is not None

        actions = th.zeros([self.valid_action_trees.shape[0], len(self.action_dims)], dtype=th.int)
        logp = th.zeros([len(self.action_dims)])
        split_dists = [VATMultiCategoricalDistribution._split_categorical(d) for d in self.distributions]
        # transpose it
        split_dists = list(map(list, zip(*split_dists)))

        for idx, (vat, dists) in enumerate(zip(self.valid_action_trees, split_dists)):
            assert isinstance(vat[0], Dict)
            action, action_logp = self._mask_and_sample_full_action(dists, vat, deterministic)
            actions[idx] = action
            logp[idx] = action_logp

        return actions

    def _mask_and_sample_full_action(self, dists: List[MaskableCategorical], vat: np.ndarray,
                                     deterministic: bool) -> Tuple[th.Tensor, th.Tensor]:
        assert isinstance(vat[0], Dict)

        action = th.zeros(len(self.action_dims), dtype=th.int)
        logp = th.zeros(len(self.action_dims))
        subtree: Dict = vat[0]
        action_type, source_node, target_node, local_vuln, remote_vuln, port, cred = range(7)
        sampled_type, sampled_type_logp, _ = self._mask_and_sample_action_part(dists[action_type], subtree,
                                                                               deterministic)
        action[action_type] = sampled_type
        logp[action_type] = sampled_type_logp

        subtree = subtree[int(sampled_type)]
        other_action_parts = {
            0: [source_node, local_vuln],
            1: [source_node, target_node, remote_vuln],
            2: [source_node, port, cred],
        }
        for part in other_action_parts[int(sampled_type)]:
            sampled, sampled_logp, _ = self._mask_and_sample_action_part(dists[part], subtree, deterministic)
            action[part] = sampled
            logp[part] = sampled_logp
            subtree = subtree[int(sampled)]

        return action, logp.sum()

    @staticmethod
    def _mask_and_sample_action_part(dist: MaskableCategorical, vat_subtree: Dict,
                                     deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, np.ndarray]:
        options = list(vat_subtree.keys())
        # mask = np.full_like(dist.logits, False, dtype=bool)
        mask = np.full(tuple(dist.logits.shape), False, dtype=bool)
        mask[options] = True
        dist.apply_masking(mask)

        if deterministic:
            sampled = th.argmax(dist.probs)
        else:
            sampled = dist.sample()
        logp = dist.log_prob(sampled)
        return sampled, logp, mask

    @staticmethod
    def _split_categorical(dist: MaskableCategorical) -> List[MaskableCategorical]:
        return [MaskableCategorical(logits=logits) for logits in dist.logits]


def make_vat_proba_distribution(action_space: spaces.Space) -> VATDistribution:
    if isinstance(action_space, spaces.MultiDiscrete):
        return VATMultiCategoricalDistribution(action_space.nvec)
    else:
        raise NotImplementedError
