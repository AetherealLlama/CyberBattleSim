from typing import Tuple, Optional

import gym
import numpy as np
from gym import spaces

import cyberbattle.agents.sb3.feature as f
from cyberbattle._env import cyberbattle_env
from cyberbattle._env.cyberbattle_env import EnvironmentBounds, CyberBattleEnv


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

    # wrapper_action_mask = wrapper_action_mask.astype(bool)
    return wrapper_action_mask


class SB3ActionModel(spaces.MultiDiscrete):
    def __init__(self, ep: EnvironmentBounds):
        self.ep = ep
        # action type x source node x target x local attack x remote attack x port x cred id
        super(SB3ActionModel, self).__init__([
            3,
            self.ep.maximum_node_count,
            self.ep.maximum_node_count,
            self.ep.local_attacks_count,
            self.ep.remote_attacks_count,
            self.ep.port_count,
            self.ep.maximum_total_credentials,
        ])

    @staticmethod
    def get_gym_action(action: np.ndarray) -> cyberbattle_env.Action:
        assert len(action) >= 7, "Invalid action vector"
        action_type = action[0]
        source_node = action[1]
        target_node = action[2]
        local_attack_id = action[3]
        remote_attack_id = action[4]
        port = action[5]
        credential_id = action[6]

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

        self._action_space = SB3ActionModel(self.ep)
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Optional[cyberbattle_env.StepInfo]]:
        action = SB3ActionModel.get_gym_action(action)
        if self.env.is_action_valid(action, None):
            obs, reward, done, info = self.env.step(action)
            self.last_obs = obs
            return self.observation(obs), reward, done, info
        else:
            return self.observation(self.last_obs), -5, False, cyberbattle_env.StepInfo(
                description='CyberBattle simulation',
                duration_in_ms=0.0,
                step_count=0,
                network_availability=1.0)

    def observation(self, obs: cyberbattle_env.Observation) -> np.ndarray:
        return self._observation_space.get(obs)

    # def is_action_valid(self, action: cyberbattle_env.Action) -> bool:
    #     action_mask: cyberbattle_env.ActionMask = self.env.compute_action_mask()
    #     if 'local_vulnerability' in action:
    #         if action_mask['local_vulnerability'][tuple(action['local_vulnerability'])] == 1:
    #             return True
    #     elif 'remote_vulnerability' in action:
    #         if action_mask['remote_vulnerability'][tuple(action['remote_vulnerability'])] == 1:
    #             return True
    #     elif 'connect' in action:
    #         if action_mask['connect'][tuple(action['connect'])] == 1:
    #             return True
    #
    #     return False
