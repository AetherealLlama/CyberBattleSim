# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A CyberBattle simulation over a randomly generated network"""

from . import cyberbattle_env
from ..simulation import generate_network


class CyberBattleRandom(cyberbattle_env.CyberBattleEnv):
    """A sample CyberBattle environment"""

    def __init__(self, n_servers_per_protocol: int = 15, n_clients: int = 50, **kwargs):
        self.n_servers_per_protocol = n_servers_per_protocol
        self.n_clients = n_clients
        super().__init__(initial_environment=generate_network.new_environment(
            n_servers_per_protocol=self.n_servers_per_protocol,
            n_clients=self.n_clients,
        ),
            maximum_discoverable_credentials_per_action=15, **kwargs)

    @property
    def name(self) -> str:
        return f'CyberBattleRandom-{self.n_servers_per_protocol}-{self.n_clients}'
