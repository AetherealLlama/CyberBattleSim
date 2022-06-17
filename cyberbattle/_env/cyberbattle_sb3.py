import random
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict

import networkx as nx
import numpy as np

from .._env import cyberbattle_env
from ..simulation import model as m
from ..simulation.model import NodeID, NodeInfo, Identifiers, CredentialID, PortName, FirewallConfiguration, \
    FirewallRule, RulePermission

ENV_IDENTIFIERS = Identifiers(
    properties=[
        'Windows',
        'Linux',
        'ApacheWebSite',
        'IIS_2019',
        'IIS_2020_patched',
        'MySql',
        'Ubuntu',
        'nginx/1.10.3',
        'SMB_vuln',
        'SMB_vuln_patched',
        'SQLServer',
        'Win10',
        'Win10Patched',
        'FLAG:Linux'
    ],
    ports=[
        'HTTPS',
        'GIT',
        'SSH',
        'RDP',
        'PING',
        'MySQL',
        'SSH-key',
        'su'
    ],
    local_vulnerabilities=[
        'ScanBashHistory',
        'ScanExplorerRecentFiles',
        'SudoAttempt',
        'CrackKeepPassX',
        'CrackKeepPass'
    ],
    remote_vulnerabilities=[
        'ProbeLinux',
        'ProbeWindows'
    ]
)


def create_random_traffic_network(size: int = 9, prob: float = 0.15, max_edges: int = 25) -> nx.DiGraph:
    g = nx.erdos_renyi_graph(size, prob, directed=True)
    while not nx.is_strongly_connected(g) and len(list(g.edges)) < max_edges:
        g = nx.erdos_renyi_graph(size, prob, directed=True)
    assert nx.is_strongly_connected(g) and len(list(g.edges)) < max_edges

    digraph = nx.DiGraph()
    ports = ['SSH', 'RDP', 'MySQL']
    for (u, v) in g.edges:
        port = np.random.choice(ports)
        digraph.add_edge(u, v, protocol=port)

    return digraph


def cyberbattle_model_from_traffic_graph(g: nx.DiGraph) -> nx.DiGraph:
    graph = nx.relabel_nodes(g, {i: str(i) for i in g.nodes})

    password_counter: int = 0

    def generate_password() -> CredentialID:
        nonlocal password_counter
        password_counter += 1
        return f'unique_pwd{password_counter}'

    def traffic_targets(source_node: NodeID, protocol: PortName) -> List[NodeID]:
        neighbors = [t for (s, t) in graph.edges()
                     if s == source_node and protocol in graph.edges[(s, t)]['protocol']]
        return neighbors

    firewall_conf = FirewallConfiguration(
        [
            FirewallRule("RDP", RulePermission.ALLOW),
            FirewallRule("SSH", RulePermission.ALLOW),
            FirewallRule("MySQL", RulePermission.ALLOW),
        ],
        [
            FirewallRule("RDP", RulePermission.ALLOW),
            FirewallRule("SSH", RulePermission.ALLOW),
            FirewallRule("MySQL", RulePermission.ALLOW),
        ],
    )

    assigned_passwords: DefaultDict[Tuple[NodeID, PortName], List[CredentialID]] = defaultdict(list)

    def assign_new_password(node: NodeID, port: PortName):
        pwd = generate_password()
        assigned_passwords[node, port].append(pwd)
        return pwd

    def create_vulnerabilities(node_id: NodeID, library: m.VulnerabilityLibrary = None) -> m.VulnerabilityLibrary:
        if library is None:
            library = {}

        rdp_neighbors = traffic_targets(node_id, 'RDP')
        if len(rdp_neighbors) > 0:
            library['ScanBashHistory'] = m.VulnerabilityInfo(
                description="Scan bash history for possible references to other machines",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node=target_node, port='RDP',
                                       credential=assign_new_password(target_node, 'RDP'))
                    for target_node in rdp_neighbors
                ]),
                reward_string="Found a reference to a remote Windows node in bash history",
                cost=1.0
            )

        ssh_neighbors = traffic_targets(node_id, 'SSH')
        if len(ssh_neighbors) > 0:
            library['ScanExplorerRecentFiles'] = m.VulnerabilityInfo(
                description="Scan Windows Explorer recent files for possible references to other machines",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node=target_node, port='SSH',
                                       credential=assign_new_password(target_node, 'SSH'))
                    for target_node in rdp_neighbors
                ]),
                reward_string="Found a reference to a remote Linux node in bash history",
                cost=1.0
            )

        sql_neighbors = traffic_targets(node_id, 'MySQL')
        if len(sql_neighbors) > 0:
            library['SudoAttempt'] = m.VulnerabilityInfo(
                description="Attempt to access MySQL root privileges",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node=target_node, port='MySQL',
                                       credential=assign_new_password(target_node, 'MySQL'))
                    for target_node in rdp_neighbors
                ]),
                reward_string="lorem ipsum",
                cost=1.0
            )

        library['ProbeLinux'] = m.VulnerabilityInfo(
            description="Probe to check if the node runs Linux",
            type=m.VulnerabilityType.REMOTE,
            outcome=m.ProbeSucceeded(["Ubuntu"]),
            reward_string="Remote machine is running Linux",
            cost=5.0
        )
        library['ProbeWindows'] = m.VulnerabilityInfo(
            description="Probe to check if the node runs Windows",
            type=m.VulnerabilityType.REMOTE,
            outcome=m.ProbeFailed(),
            reward_string="Remote machine is not running Windows",
            cost=5.0
        )

        return library

    entry_neighbor_index = random.randrange(len(graph.nodes))
    entry_neighbor_id, entry_node_data = list(graph.nodes(data=True))[entry_neighbor_index]
    entry_node_index = len(graph.nodes)
    graph.add_node(str(entry_node_index))
    graph.nodes[str(entry_node_index)].update(
        {
            'data': m.NodeInfo(
                services=[],
                value=0,
                vulnerabilities=dict(
                    ScanExplorerRecentFiles=m.VulnerabilityInfo(
                        description="Scan Windows Explorer recent files for possible references to other machines",
                        type=m.VulnerabilityType.LOCAL,
                        outcome=m.LeakedCredentials(credentials=[
                            m.CachedCredential(node=entry_neighbor_id, port="SSH",
                                               credential=assign_new_password(entry_neighbor_id, "SSH")),
                        ]),
                        reward_string="Found a reference to a remote Linux node in bash history",
                        cost=1.0,
                    )
                ),
                agent_installed=True,
                reimagable=False,
            )
        }
    )

    def create_node_data(node_id: NodeID):
        return m.NodeInfo(
            services=[m.ListeningService(name=port, allowedCredentials=assigned_passwords[(target_node, port)])
                      for (target_node, port) in assigned_passwords.keys()
                      if target_node == node_id],
            value=random.randint(100, 200),
            vulnerabilities=create_vulnerabilities(node_id),
            agent_installed=False,
            firewall=firewall_conf,
        )

    for node in list(graph.nodes):
        if node != str(entry_node_index):
            graph.nodes[node].clear()
            graph.nodes[node].update({'data': create_node_data(node)})

    return graph


def create_network(size: int, prob: float = 0.15) -> nx.DiGraph:
    g = create_random_traffic_network(size, prob)
    network = cyberbattle_model_from_traffic_graph(g)

    return network


def new_environment(size: int, prob: float = 0.15) -> m.Environment:
    return m.Environment(
        network=create_network(size, prob),
        vulnerability_library=dict([]),
        identifiers=ENV_IDENTIFIERS,
    )


class CyberBattleSB3(cyberbattle_env.CyberBattleEnv):
    def __init__(self, size: int, prob: float, **kwargs):
        self.size = size
        self.prob = prob
        super().__init__(initial_environment=new_environment(self.size), **kwargs)

    @property
    def name(self) -> str:
        return f'CyberBattleSB3-{self.size}'
