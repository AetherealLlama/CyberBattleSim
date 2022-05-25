from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gym import spaces

from cyberbattle._env import cyberbattle_env
from cyberbattle._env.cyberbattle_env import EnvironmentBounds


def owned_nodes(obs: cyberbattle_env.Observation):
    return np.nonzero(obs['nodes_privilegelevel'])[0]


def discovered_nodes_notowned(obs: cyberbattle_env.Observation):
    return np.nonzero(obs['nodes_privilegelevel'] == 0)[0]


class Feature(spaces.MultiDiscrete, ABC):
    def __init__(self, ep: EnvironmentBounds, nvec):
        self.ep = ep
        super().__init__(nvec)

    @property
    def flat_size(self):
        return np.prod(self.nvec)

    @abstractmethod
    def get(self, obs: cyberbattle_env.Observation) -> np.ndarray:
        raise NotImplementedError


class ConcatFeatures(Feature):
    def __init__(self, ep: EnvironmentBounds, features: List[Feature]):
        self.features = features
        self.dim_sizes = np.concatenate([f.nvec for f in self.features])
        super().__init__(ep, self.dim_sizes)

    def get(self, obs: cyberbattle_env.Observation) -> np.ndarray:
        feature_vector = [f.get(obs) for f in self.features]
        feature_vector = np.concatenate(feature_vector)
        # feature_vector = np.expand_dims(feature_vector, 0)
        return feature_vector


class FeatureGlobalNodesProperties(Feature):
    def __init__(self, ep: EnvironmentBounds):
        super(FeatureGlobalNodesProperties, self).__init__(ep, [4] * ep.property_count * ep.maximum_node_count)

    def get(self, obs: cyberbattle_env.Observation) -> np.ndarray:
        features = []
        for i in range(self.ep.maximum_node_count):
            if i < len(obs['discovered_nodes_properties']):
                features.append(np.copy(obs['discovered_nodes_properties'][i]) + 1)
            else:
                features.append(np.ones(self.ep.property_count))
        return np.concatenate(features)


class FeatureGlobalCredentialCacheLength(Feature):
    def __init__(self, ep: EnvironmentBounds):
        super().__init__(ep, [ep.maximum_total_credentials])

    def get(self, obs: cyberbattle_env.Observation) -> np.ndarray:
        return np.array([obs['credential_cache_length']])


class FeatureGlobalCredentialCache(Feature):
    def __init__(self, ep: EnvironmentBounds):
        super(FeatureGlobalCredentialCache, self).__init__(ep, [ep.maximum_node_count,
                                                                ep.port_count] * ep.maximum_total_credentials)

    def get(self, obs: cyberbattle_env.Observation) -> np.ndarray:
        features = [
            obs['credential_cache_matrix'][i] if i < len(obs['credential_cache_matrix']) else np.zeros(2)
            for i in range(self.ep.maximum_total_credentials)
        ]
        return np.concatenate(features)


class FeatureGlobalNodesPrivilegeLevel(Feature):
    def __init__(self, ep: EnvironmentBounds, max_privilege_level: int):
        self.max_privilege_level = max_privilege_level
        super(FeatureGlobalNodesPrivilegeLevel, self).__init__(ep, [max_privilege_level + 1] * ep.maximum_node_count)

    def get(self, obs: cyberbattle_env.Observation) -> np.ndarray:
        features = np.array(obs['nodes_privilegelevel']) + 1
        features.resize(self.ep.maximum_node_count, refcheck=False)
        return features
