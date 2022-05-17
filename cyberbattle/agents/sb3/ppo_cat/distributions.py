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
        action = actions[0]
        subtree = self.valid_action_trees[0][0]
        action_type, source_node, target_node, local_vuln, remote_vuln, port, cred = range(7)

        logp_parts = th.zeros([7])
        entropy_parts = th.zeros([7])

        sampled_type = action[action_type]
        subtree = subtree[int(sampled_type)]
        if sampled_type == 0:
            for part in [source_node, local_vuln]:
                logp, entropy = self._logp_and_entropy_for_action_part(subtree, action[part], part)
                logp_parts[part] = logp
                entropy_parts[part] = entropy
                subtree = subtree[int(action[part])]
        elif sampled_type == 1:
            for part in [source_node, target_node, remote_vuln]:
                logp, entropy = self._logp_and_entropy_for_action_part(subtree, action[part], part)
                logp_parts[part] = logp
                entropy_parts[part] = entropy
                subtree = subtree[int(action[part])]
        else:
            for part in [source_node, port, cred]:
                logp, entropy = self._logp_and_entropy_for_action_part(subtree, action[part], part)
                logp_parts[part] = logp
                entropy_parts[part] = entropy
                subtree = subtree[int(action[part])]
        return logp_parts.sum(), entropy_parts.sum()

    def _logp_and_entropy_for_action_part(self, subtree: Dict, sampled: th.Tensor, dist_pos: int) -> Tuple[th.Tensor, th.Tensor]:
        options = list(subtree.keys())
        mask = np.full_like(self.distributions[dist_pos].logits[0], False, dtype=bool)
        mask[options] = True
        self.distributions[dist_pos].apply_masking(mask)

        logp = self.distributions[dist_pos].log_prob(sampled)
        entropy = self.distributions[dist_pos].entropy()
        return logp, entropy

    def entropy(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1)

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

    def _mask_and_sample(self, deterministic: bool = False) -> th.Tensor:
        assert self.valid_action_trees is not None
        assert isinstance(self.valid_action_trees[0][0], Dict)

        subtree = self.valid_action_trees[0][0]
        action = th.zeros([7], dtype=th.int)
        action_type, source_node, target_node, local_vuln, remote_vuln, port, cred = range(7)
        sampled_type = self._mask_and_sample_action_part(subtree, action_type, deterministic)
        action[action_type] = sampled_type
        subtree = subtree[int(sampled_type)]

        if sampled_type == 0:
            for part in [source_node, local_vuln]:
                sampled = self._mask_and_sample_action_part(subtree, part, deterministic)
                action[part] = sampled
                subtree = subtree[int(sampled)]
        elif sampled_type == 1:
            for part in [source_node, target_node, remote_vuln]:
                sampled = self._mask_and_sample_action_part(subtree, part, deterministic)
                action[part] = sampled
                subtree = subtree[int(sampled)]
        else:
            for part in [source_node, port, cred]:
                sampled = self._mask_and_sample_action_part(subtree, part, deterministic)
                action[part] = sampled
                subtree = subtree[int(sampled)]
        return action.unsqueeze(0)

    def _mask_and_sample_action_part(
        self,
        subtree: Dict,
        dist_pos: int,
        deterministic: bool
    ) -> th.Tensor:
        options = list(subtree.keys())
        mask = np.full_like(self.distributions[dist_pos].logits[0], False, dtype=bool)
        mask[options] = True
        self.distributions[dist_pos].apply_masking(mask)

        if deterministic:
            sampled = th.argmax(self.distributions[dist_pos].probs)
        else:
            sampled = self.distributions[dist_pos].sample()
        return sampled


# class VATMultiCategoricalDistribution(VATDistribution):
#     def __init__(self, action_dims: List[int]):
#         super().__init__()
#         self.distributions: List[MaskableCategorical] = []
#         self.action_dims = action_dims
#
#     def proba_distribution_net(self, latent_dim: int) -> nn.Module:
#         action_logits = nn.Linear(latent_dim, sum(self.action_dims))
#         return action_logits
#
#     def proba_distribution(self, action_logits: th.Tensor) -> "VATMultiCategoricalDistribution":
#         reshaped_logits = action_logits.view(-1, sum(self.action_dims))
#
#         self.distributions = [
#             MaskableCategorical(logits=split) for split in th.split(reshaped_logits, tuple(self.action_dims), dim=1)
#         ]
#         return self
#
#     def log_prob(self, actions: th.Tensor) -> th.Tensor:
#         assert len(self.distributions) > 0, "Must set distribution parameters"
#
#         # Restructure shape to align with each categorical
#         actions = actions.view(-1, len(self.action_dims))
#
#         # Extract each discrete action and compute log prob for their respective distributions
#         return th.stack(
#             [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1))], dim=1
#         ).sum(dim=1)
#
#     def entropy(self) -> th.Tensor:
#         assert len(self.distributions) > 0, "Must set distribution parameters"
#         return th.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1)
#
#     def sample(self, valid_action_trees: Optional[np.ndarray] = None) -> th.Tensor:
#         assert len(self.distributions) > 0, "Must set distribution parameters"
#         sampled, _ = self._mask_and_sample(valid_action_trees, False)
#         return sampled
#
#     def mode(self, valid_action_trees: Optional[np.ndarray] = None) -> th.Tensor:
#         assert len(self.distributions) > 0, "Must set distribution parameters"
#         sampled, _ = self._mask_and_sample(valid_action_trees, True)
#         return sampled
#
#     def get_actions(self, deterministic: bool = False, valid_action_trees: Optional[np.ndarray] = None) -> th.Tensor:
#         if deterministic:
#             return self.mode(valid_action_trees)
#         return self.sample(valid_action_trees)
#
#     def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         # Update the proba distribution
#         self.proba_distribution(action_logits)
#         return self.get_actions(deterministic=deterministic)
#
#     def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         actions = self.actions_from_params(action_logits)
#         log_prob = self.log_prob(actions)
#         return actions, log_prob
#
#     def apply_vat(self, vat: Optional[np.ndarray]) -> None:
#         pass
#
#     def logp_entropy_for_action(self, action: th.Tensor, valid_action_tree: Optional[np.ndarray]) -> Tuple[th.Tensor, th.Tensor]:
#         action_type, source_node, target_node, local_vuln, remote_vuln, port, cred = range(7)
#
#         logp_parts = th.zeros([7])
#         entropy_parts = th.zeros([7])
#
#         sampled_type = action[action_type]
#         subtree = valid_action_tree[sampled_type]
#         if sampled_type == 0:
#             for part in [source_node, local_vuln]:
#                 options = list(subtree.keys())
#                 mask = np.full_like(self.distributions[part].logits, False, dtype=bool)
#                 mask[options] = True
#                 self.distributions[part].apply_masking(mask)
#                 logp_parts[part] = self.distributions[part].log_prob(action[part])
#                 entropy_parts[part] = self.distributions[part].entropy()
#         elif sampled_type == 1:
#             for part in [source_node, target_node, remote_vuln]:
#                 options = list(subtree.keys())
#                 mask = np.full_like(self.distributions[part].logits, False, dtype=bool)
#                 mask[options] = True
#                 self.distributions[part].apply_masking(mask)
#                 logp_parts[part] = self.distributions[part].log_prob(action[part])
#                 entropy_parts[part] = self.distributions[part].entropy()
#         elif sampled_type == 2:
#             for part in [source_node, port, cred]:
#                 options = list(subtree.keys())
#                 mask = np.full_like(self.distributions[part].logits, False, dtype=bool)
#                 mask[options] = True
#                 self.distributions[part].apply_masking(mask)
#                 logp_parts[part] = self.distributions[part].log_prob(action[part])
#                 entropy_parts[part] = self.distributions[part].entropy()
#         else:
#             raise ValueError
#         return logp_parts.sum(), entropy_parts.sum()
#
#     def _mask_and_sample(self, valid_action_tree: Optional[np.ndarray], deterministic: bool = False) -> Tuple[
#         th.Tensor, th.Tensor]:
#         assert len(self.distributions) > 0, "Must set distribution params"
#         if valid_action_tree is None:
#             if deterministic:
#                 actions = self.mode()
#             else:
#                 actions = self.sample()
#             logp = self.log_prob(actions)
#             return actions, logp
#         assert isinstance(valid_action_tree[0][0], Dict)
#         vat: Dict = valid_action_tree[0][0]
#
#         # this is a shitshow of hardcoded values
#         action = th.zeros([7])
#         logp_parts = th.zeros([7])
#
#         action_type, source_node, target_node, local_vuln, remote_vuln, port, cred = range(7)
#         sampled_type, sampled_type_logp = self._mask_and_sample_action_part(vat, action_type)
#         logp_parts[action_type] = sampled_type_logp
#         action[action_type] = sampled_type
#         subtree = vat[sampled_type]
#
#         if sampled_type == 0:
#             for part in [source_node, local_vuln]:
#                 sampled, sampled_logp = self._mask_and_sample_action_part(part, subtree, deterministic)
#                 logp_parts[part] = sampled_logp
#                 action[part] = sampled
#                 subtree = subtree[sampled]
#         elif sampled_type == 1:
#             for part in [source_node, target_node, remote_vuln]:
#                 sampled, sampled_logp = self._mask_and_sample_action_part(part, subtree, deterministic)
#                 logp_parts[part] = sampled_logp
#                 action[part] = sampled
#                 subtree = subtree[sampled]
#         elif sampled_type == 2:
#             for part in [source_node, port, cred]:
#                 sampled, sampled_logp = self._mask_and_sample_action_part(part, subtree, deterministic)
#                 logp_parts[part] = sampled_logp
#                 action[part] = sampled
#                 subtree = subtree[sampled]
#         else:
#             raise ValueError
#         return action, logp_parts.sum()
#
#     def _mask_and_sample_action_part(
#         self,
#         vat_subtree: Dict,
#         dist_pos: int, deterministic: bool
#     ) -> Tuple[th.Tensor, th.Tensor]:
#         options = list(vat_subtree.keys())
#         mask = np.full_like(self.distributions[dist_pos].logits, False, dtype=bool)
#         mask[options] = True
#         self.distributions[dist_pos].apply_masking(mask)
#
#         if deterministic:
#             sampled = th.argmax(self.distributions[dist_pos].probs)
#         else:
#             sampled = self.distributions[dist_pos].sample()
#         logp = self.distributions[dist_pos].log_prob(sampled)
#         return sampled, logp


def make_vat_proba_distribution(action_space: spaces.Space) -> VATDistribution:
    if isinstance(action_space, spaces.MultiDiscrete):
        return VATMultiCategoricalDistribution(action_space.nvec)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Discrete, MultiDiscrete."
        )
