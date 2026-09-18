"""
network.graph
--------------

The internal graph representation (architecture document Part IV §4.3:
Network -> Graph -> DAE -> ODE -> IMS). A `NetworkGraph` turns a `Network`'s
bus/branch lists into an adjacency structure and a signed incidence
matrix, which is the natural form for both topology-level queries (which
branches touch a bus, is the network connected) and for mechanically
assembling Kirchhoff's current law in `network.builder`.

Sign convention: branch current i_b is defined positive flowing from
`from_bus` to `to_bus`. The incidence matrix entry A[bus_index, branch_index]
is +1 if the branch's `to_bus` is that bus (current flows IN), -1 if the
branch's `from_bus` is that bus (current flows OUT), 0 otherwise -- so
`A @ i_branches` gives, for every bus, the net branch current flowing IN.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from .topology import Network, Branch


class NetworkGraph:
    """Graph representation of a Network's topology, built once and reused by the Automatic Model Builder."""

    def __init__(self, network: Network):
        self.network = network
        self.bus_ids: List[str] = [b.id for b in network.buses]
        self.branch_ids: List[str] = [br.id for br in network.branches]
        self._bus_pos: Dict[str, int] = {bid: i for i, bid in enumerate(self.bus_ids)}
        self._branch_pos: Dict[str, int] = {bid: i for i, bid in enumerate(self.branch_ids)}

        self.adjacency: Dict[str, List[Tuple[Branch, str]]] = {bid: [] for bid in self.bus_ids}
        for br in network.branches:
            self.adjacency[br.from_bus].append((br, br.to_bus))
            self.adjacency[br.to_bus].append((br, br.from_bus))

        self.incidence = self._build_incidence()

    def _build_incidence(self) -> np.ndarray:
        n_bus, n_branch = len(self.bus_ids), len(self.branch_ids)
        A = np.zeros((n_bus, n_branch))
        for j, br in enumerate(self.network.branches):
            A[self._bus_pos[br.to_bus], j] += 1.0
            A[self._bus_pos[br.from_bus], j] -= 1.0
        return A

    def branches_at(self, bus_id: str) -> List[Tuple[Branch, str]]:
        """Branches touching bus_id, as (branch, other_bus_id) pairs."""
        return self.adjacency[bus_id]

    def bus_index(self, bus_id: str) -> int:
        return self._bus_pos[bus_id]

    def branch_index(self, branch_id: str) -> int:
        return self._branch_pos[branch_id]

    def is_connected(self) -> bool:
        if not self.bus_ids:
            return True
        seen = set()
        stack = [self.bus_ids[0]]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(other for _, other in self.adjacency[cur])
        return len(seen) == len(self.bus_ids)
