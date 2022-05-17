from typing import Tuple, Optional, Dict, Any

import gym
import numpy as np
import torch
from gym import spaces

import cyberbattle.agents.sb3.feature as f
from cyberbattle._env import cyberbattle_env
from cyberbattle._env.cyberbattle_env import EnvironmentBounds, CyberBattleEnv, Observation


def mask_fn(env: gym.Env) -> np.ndarray:
    action_mask: cyberbattle_env.ActionMask = env.compute_action_mask()
    ep: EnvironmentBounds = env.bounds

    wrapper_action_mask = np.zeros((3, ep.maximum_node_count, ep.maximum_node_count, ep.local_attacks_count,
                                    ep.remote_attacks_count, ep.port_count, ep.maximum_total_credentials))
    for source_node, x in enumerate(action_mask['local_vulnerability']):
        for vuln_id, _ in enumerate(x):
            if action_mask['local_vulnerability'][source_node, vuln_id] == 1:
                wrapper_action_mask[0, source_node, :, vuln_id, :, :, :] = 1.0

    for source_node, x in enumerate(action_mask['remote_vulnerability']):
        for target_node, y in enumerate(x):
            for vuln_id, _ in enumerate(y):
                if action_mask['remote_vulnerability'][source_node, target_node, vuln_id] == 1:
                    wrapper_action_mask[1, source_node, target_node, :, vuln_id, :, :] = 1.0

    for source_node, x in enumerate(action_mask['connect']):
        for target_node, y in enumerate(x):
            for port_id, z in enumerate(y):
                for cred_id, _ in enumerate(z):
                    if action_mask['connect'][source_node, target_node, port_id, cred_id] == 1:
                        wrapper_action_mask[2, source_node, target_node, :, :, port_id, cred_id] = 1.0

    wrapper_action_mask = wrapper_action_mask.astype(bool)
    return wrapper_action_mask


def mask_fn_2(env: gym.Env) -> np.ndarray:
    obs: Observation = env.last_observation
    ep: EnvironmentBounds = env.bounds

    action_type = np.array([True] * 3)

    source_nodes = np.array([False] * ep.maximum_node_count)
    source_nodes[f.owned_nodes(obs)] = True

    target_nodes = np.concatenate((np.array([True] * obs['discovered_node_count']),
                                   np.array([False] * (ep.maximum_node_count - obs['discovered_node_count']))))

    local_attack = np.array([True] * ep.local_attacks_count)
    remote_attack = np.array([True] * ep.remote_attacks_count)
    ports = np.array([True] * ep.port_count)

    credentials = np.concatenate((np.array([True] * obs['credential_cache_length']),
                                  np.array([False] * (ep.maximum_total_credentials - obs['credential_cache_length']))))

    return np.concatenate((action_type, source_nodes, target_nodes, local_attack,
                           remote_attack, ports, credentials))


class SB3MultiDiscreteActionModel(spaces.MultiDiscrete):
    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep
        # action type x source node x target x local attack x remote attack x port x cred id
        self.nvec = (
            3,
            self.ep.maximum_node_count,
            self.ep.maximum_node_count,
            self.ep.local_attacks_count,
            self.ep.remote_attacks_count,
            self.ep.port_count,
            self.ep.maximum_total_credentials,
        )
        super().__init__(self.nvec)

    def get_gym_action(self, action: np.ndarray) -> cyberbattle_env.Action:
        assert isinstance(action, np.ndarray), "action must be np array"
        assert len(action) >= 7, "Invalid action vector"
        action_type, source_node, target_node, local_attack_id, remote_attack_id, port, credential_id = action

        if action_type == 0:
            return {'local_vulnerability': np.array([source_node, local_attack_id])}
        elif action_type == 1:
            return {'remote_vulnerability': np.array([source_node, target_node, remote_attack_id])}
        elif action_type == 2:
            return {'connect': np.array([source_node, target_node, port, credential_id])}
        else:
            raise ValueError


class SB3DiscreteActionModel(spaces.Discrete):
    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep

        # action type x source node x target x local attack x remote attack x port x cred id
        self.nvec = (
            3,
            self.ep.maximum_node_count,
            self.ep.maximum_node_count,
            self.ep.local_attacks_count,
            self.ep.remote_attacks_count,
            self.ep.port_count,
            self.ep.maximum_total_credentials,
        )
        super().__init__(np.prod(self.nvec))

        self.mapping = tuple(np.ndindex(self.nvec))

    def get_gym_action(self, action: int) -> cyberbattle_env.Action:
        assert 0 <= action < np.prod(self.nvec)

        action_vec = self.mapping[action]
        action_type, source_node, target_node, local_attack_id, remote_attack_id, port, credential_id = action_vec

        if action_type == 0:
            return {'local_vulnerability': np.array([source_node, local_attack_id])}
        elif action_type == 1:
            return {'remote_vulnerability': np.array([source_node, target_node, remote_attack_id])}
        elif action_type == 2:
            return {'connect': np.array([source_node, target_node, port, credential_id])}
        else:
            raise ValueError


class SB3EnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, ep: EnvironmentBounds):
        super(SB3EnvWrapper, self).__init__(env)

        self.ep = ep

        self._action_space = SB3MultiDiscreteActionModel(self.ep)
        # self._action_space = SB3DiscreteActionModel(self.ep)
        self._observation_space = f.ConcatFeatures(self.ep, [
            f.FeatureGlobalNodesProperties(ep),
            f.FeatureGlobalCredentialCacheLength(ep),
            f.FeatureGlobalCredentialCache(ep),
            f.FeatureGlobalNodesPrivilegeLevel(ep, CyberBattleEnv.privilege_levels)
        ])

        self.last_obs = None

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.last_obs = obs
        return self.observation(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Optional[cyberbattle_env.StepInfo]]:
        # action = SB3ActionModel.get_gym_action(action)
        action = self.action_space.get_gym_action(action)
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs: cyberbattle_env.Observation) -> np.ndarray:
        return self._observation_space.get(obs)

    def valid_action_trees(self) -> np.ndarray:
        action_mask: cyberbattle_env.ActionMask = self.env.compute_action_mask()
        valid_action_trees = {}

        # this is ugly as fuck innit
        local_vulnerabilities = np.transpose((action_mask['local_vulnerability'] > 0).nonzero())
        if len(local_vulnerabilities) > 0:
            local_vulnerabilities_tree = {}
            for i in local_vulnerabilities:
                source_node, vuln_id = i
                if source_node not in local_vulnerabilities_tree:
                    local_vulnerabilities_tree[source_node] = {}
                local_vulnerabilities_tree[source_node][vuln_id] = {}
            valid_action_trees[0] = local_vulnerabilities_tree

        remote_vulnerabilities = np.transpose((action_mask['remote_vulnerability'] > 0).nonzero())
        if len(remote_vulnerabilities) > 0:
            remote_vulnerabilities_tree = {}
            for i in remote_vulnerabilities:
                source_node, target_node, vuln_id = i
                if source_node not in remote_vulnerabilities_tree:
                    remote_vulnerabilities_tree[source_node] = {}
                if target_node not in remote_vulnerabilities_tree[source_node]:
                    remote_vulnerabilities_tree[source_node][target_node] = {}
                remote_vulnerabilities_tree[source_node][target_node][vuln_id] = {}
            valid_action_trees[1] = remote_vulnerabilities_tree

        connect = np.transpose((action_mask['connect'] > 0).nonzero())
        if len(connect) > 0:
            connect_tree = {}
            for i in connect:
                source_node, target_node, port_id, creds = i
                if source_node not in connect_tree:
                    connect_tree[source_node] = {}
                if target_node not in connect_tree[source_node]:
                    connect_tree[source_node][target_node] = {}
                if port_id not in connect_tree[source_node][target_node]:
                    connect_tree[source_node][target_node][port_id] = {}
                connect_tree[source_node][target_node][port_id][creds] = {}
            valid_action_trees[2] = connect_tree

        return np.array([valid_action_trees])
        # return vat_to_array(valid_action_trees)


def vat_to_array(vat: Dict) -> np.ndarray:
    ret = []
    if 0 in vat:
        local_vulnerabilities = vat[0]
        for source_node, i in local_vulnerabilities.items():
            for vuln_id in i.keys():
                ret.append([0, source_node, 0, vuln_id, 0, 0, 0])

    if 1 in vat:
        remote_vulnerabilities = vat[1]
        for source_node, i in remote_vulnerabilities.items():
            for target_node, j in i.items():
                for vuln_id in j.keys():
                    ret.append([1, source_node, target_node, 0, vuln_id, 0, 0])

    if 2 in vat:
        connect = vat[2]
        for source_node, i in connect.items():
            for target_node, j in i.items():
                for port, l in j.items():
                    for cred in l.keys():
                        ret.append([2, source_node, target_node, 0, 0, port, cred])
    ret = np.array(ret)
    # ret = ret.flatten()
    return ret


def array_to_vat(arr: np.ndarray) -> Dict:
    # arr = arr.reshape(int(len(arr) / 7), 7)
    local_vulnerabilities = [a for a in arr if a[0] == 0]
    remote_vulnerabilities = [a for a in arr if a[0] == 1]
    connect = [a for a in arr if a[0] == 2]
    ret = {}
    if len(local_vulnerabilities) > 0:
        local_vulnerabilities_tree = {}
        for i in local_vulnerabilities:
            _, source_node, _, vuln_id, _, _, _ = i
            if source_node not in local_vulnerabilities_tree:
                local_vulnerabilities_tree[source_node] = {}
            local_vulnerabilities_tree[source_node][vuln_id] = {}
        ret[0] = local_vulnerabilities_tree

    if len(remote_vulnerabilities) > 0:
        remote_vulnerabilities_tree = {}
        for i in remote_vulnerabilities:
            _, source_node, target_node, _, vuln_id, _, _ = i
            if source_node not in remote_vulnerabilities_tree:
                remote_vulnerabilities_tree[source_node] = {}
            if target_node not in remote_vulnerabilities_tree[source_node]:
                remote_vulnerabilities_tree[source_node][target_node] = {}
            remote_vulnerabilities_tree[source_node][target_node][vuln_id] = {}
        ret[1] = remote_vulnerabilities_tree

    if len(connect) > 0:
        connect_tree = {}
        for i in connect:
            _, source_node, target_node, _, _, port_id, creds = i
            if source_node not in connect_tree:
                connect_tree[source_node] = {}
            if target_node not in connect_tree[source_node]:
                connect_tree[source_node][target_node] = {}
            if port_id not in connect_tree[source_node][target_node]:
                connect_tree[source_node][target_node][port_id] = {}
            connect_tree[source_node][target_node][port_id][creds] = {}
        ret[2] = connect_tree

    return ret
