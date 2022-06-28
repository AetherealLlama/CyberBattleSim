import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import networkx as nx
import numpy as np

from .cyberbattle_env import Observation
from .._env import cyberbattle_env
from ..simulation import model as m
from ..simulation.model import NodeID, Identifiers, CredentialID, PortName, FirewallConfiguration, \
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
    ],
    ports=[
        'HTTPS',
        'GIT',
        'SSH',
        'FTP',
        'PING',
        'MySQL',
        'SSH-key',
        'su',
        'FTP'
    ],
    local_vulnerabilities=[
        'ScanBashHistory',
        'ScanExplorerRecentFiles',
        'SudoAttempt',
        'CrackKeepPassX',
        'CrackKeepPass',
        'FTPDirectoryTraversal',
        'BruteforceMySQLPassword'
    ],
    remote_vulnerabilities=[
        'ProbeLinux',
        'ProbeWindows'
    ]
)

password_counter = 0


def generate_password() -> CredentialID:
    global password_counter
    password: CredentialID = f'unique_pwd{password_counter}'
    password_counter += 1
    return password


def create_random_traffic_network(size: int = 9, prob: float = 0.15, max_edges: int = 25, seed: Optional[int] = None) -> nx.DiGraph:
    # from rich import print
    # print(f'starting network seed {seed}')

    g = nx.erdos_renyi_graph(size, prob, directed=True, seed=seed)
    while not nx.is_strongly_connected(g) or len(list(g.edges)) > max_edges:
        if seed is not None:
            seed *= 2
        g = nx.erdos_renyi_graph(size, prob, directed=True, seed=seed)
    assert nx.is_strongly_connected(g) and len(list(g.edges)) < max_edges

    # print(f'generated network with seed {seed}')

    digraph = nx.DiGraph()
    ports = ['SSH', 'FTP', 'MySQL']
    for (u, v) in g.edges:
        port = np.random.choice(ports)
        digraph.add_edge(u, v, protocol=port)

    return digraph


def traffic_neighbors(g: nx.DiGraph, source_node: NodeID, protocol: PortName) -> List[NodeID]:
    neighbors = [t for (s, t) in g.edges()
                 if s == source_node and protocol == g.edges[(s, t)]['protocol']]
    return neighbors


def assign_passwords(g: nx.DiGraph) -> Tuple[Dict, Dict]:
    password_cache: Dict[NodeID, Dict[PortName, List[CredentialID]]] = defaultdict(lambda: defaultdict(list))
    password_leaks: Dict[NodeID, Dict[PortName, List[Tuple[NodeID, CredentialID]]]] \
        = defaultdict(lambda: defaultdict(list))

    ports = ['SSH', 'FTP', 'MySQL']

    for source_node in list(g.nodes):
        for port in ports:
            neighbors = traffic_neighbors(g, source_node, port)
            if len(neighbors) > 0:
                for target_node in neighbors:
                    password = generate_password()
                    password_cache[target_node][port].append(password)
                    password_leaks[source_node][port].append((target_node, password))

    return password_cache, password_leaks


def make_local_vulns(source_node: NodeID, password_leaks: Dict) -> m.VulnerabilityLibrary:
    library: m.VulnerabilityLibrary = {}
    node_leaks = password_leaks[source_node]

    ports_vuln_names = {
        'SSH': 'ScanBashHistory',
        'FTP': 'FTPDirectoryTraversal',
        'MySQL': 'BruteforceMySQLPassword',
    }

    for port in ports_vuln_names.keys():
        if port in node_leaks:
            library[ports_vuln_names[port]] = m.VulnerabilityInfo(
                description=f"{port} local vulnerability",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node=target_node, port=port, credential=password)
                    for (target_node, password) in node_leaks[port]
                ]),
                reward_string=f'Discovered {port} creds',
                cost=2.0,
            )

    return library


def create_lateral_moves(g: nx.DiGraph) -> nx.DiGraph:
    password_cache, password_leaks = assign_passwords(g)

    firewall_conf = FirewallConfiguration(
        [
            FirewallRule("FTP", RulePermission.ALLOW),
            FirewallRule("SSH", RulePermission.ALLOW),
            FirewallRule("MySQL", RulePermission.ALLOW),
        ],
        [
            FirewallRule("FTP", RulePermission.ALLOW),
            FirewallRule("SSH", RulePermission.ALLOW),
            FirewallRule("MySQL", RulePermission.ALLOW),
        ],
    )

    for node in list(g.nodes):
        services: List[m.ListeningService] = []
        for port in password_cache[node].keys():
            service = m.ListeningService(port, password_cache[node][port])
            services.append(service)

        g.nodes[node].clear()
        g.nodes[node].update({
            'data': m.NodeInfo(
                services=services,
                value=random.randint(25, 100),
                vulnerabilities=make_local_vulns(node, password_leaks),
                firewall=firewall_conf,
            )
        })

    return g


def create_node_data(g: nx.DiGraph, node_id: NodeID) -> nx.DiGraph:
    # add some random vulns
    data: m.NodeInfo = g.nodes[node_id]['data']

    is_linux = random.random() < 0.5
    if is_linux:
        data.properties.append('Linux')
        data.properties.append('Ubuntu')
        data.vulnerabilities['ProbeLinux'] = m.VulnerabilityInfo(
            description='Probe if Linux',
            type=m.VulnerabilityType.REMOTE,
            outcome=m.ProbeSucceeded(["Linux", "Ubuntu"]),
            reward_string='Machine is running Linux',
            cost=5.0,
        )
    else:
        data.properties.append('Windows')
        data.properties.append('Win10')
        data.vulnerabilities['ProbeWindows'] = m.VulnerabilityInfo(
            description='Probe if Windows',
            type=m.VulnerabilityType.REMOTE,
            outcome=m.ProbeSucceeded(["Windows", "Win10"]),
            reward_string='Machine is running Windows',
            cost=5.0,
        )

    return g


def cyberbattle_model_from_traffic_graph(g: nx.DiGraph) -> nx.DiGraph:
    graph = nx.relabel_nodes(g, {i: str(i) for i in g.nodes})

    graph = create_lateral_moves(graph)

    entry_node_index = random.randrange(len(graph.nodes))
    entry_node_id, entry_node_data = list(graph.nodes(data=True))[entry_node_index]

    entry_node_data['data'].agent_installed = True
    entry_node_data['data'].reimagable = False

    for node in list(graph.nodes):
        create_node_data(graph, node)

    return graph


def create_network(size: int, prob: float = 0.15, seed: Optional[int] = None) -> nx.DiGraph:
    g = create_random_traffic_network(size, prob, seed=seed)
    network = cyberbattle_model_from_traffic_graph(g)

    return network


def new_environment(size: int, prob: float = 0.15, seed: Optional[int] = None) -> m.Environment:
    return m.Environment(
        network=create_network(size, prob, seed=seed),
        vulnerability_library=dict([]),
        identifiers=ENV_IDENTIFIERS,
    )


class CyberBattleSB3(cyberbattle_env.CyberBattleEnv):
    def __init__(self, size: int, prob: float = 0.15, seed: Optional[int] = None, **kwargs):
        self.size = size
        self.prob = prob
        self.seed = seed
        super().__init__(initial_environment=new_environment(self.size, self.prob, self.seed), **kwargs)

    @property
    def name(self) -> str:
        return f'CyberBattleSB3-{self.size}'

    def reset(self, regen_env: bool = True) -> Observation:
        if regen_env:
            new_env = new_environment(self.size, self.prob, self.seed)
            self.install_new_environment(new_env)
        return super().reset()
