"""Minimal drop-in replacement for the slice of the networkx API this module uses.

networkx is listed in requirements.txt and is used when available; this fallback
keeps the pipeline runnable on a bare Python install (and keeps the graph code
independent of any external graph library).
"""
import numpy as np


class _NodeView:
    def __init__(self, g):
        self._g = g

    def __call__(self, data=False):
        if data:
            return list(self._g._node.items())
        return list(self._g._node.keys())

    def __iter__(self):
        return iter(self._g._node)

    def __getitem__(self, n):
        return self._g._node[n]

    def __contains__(self, n):
        return n in self._g._node

    def __len__(self):
        return len(self._g._node)


class Graph:
    def __init__(self):
        self._node = {}
        self._adj = {}

    # ---- construction ----
    def add_node(self, n, **attrs):
        if n not in self._node:
            self._node[n] = {}
            self._adj[n] = {}
        self._node[n].update(attrs)

    def add_edge(self, u, v, **attrs):
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = attrs
        self._adj[v][u] = attrs

    # ---- access ----
    @property
    def nodes(self):
        return _NodeView(self)

    def edges(self, data=False):
        seen, out = set(), []
        for u, nbrs in self._adj.items():
            for v, attrs in nbrs.items():
                key = (v, u) if (v, u) in seen else (u, v)
                if key in seen:
                    continue
                seen.add((u, v))
                out.append((u, v, attrs) if data else (u, v))
        return out

    def degree(self, nbunch=None):
        if nbunch is None:
            return [(n, len(self._adj[n])) for n in self._node]
        return len(self._adj[nbunch])

    def neighbors(self, n):
        return iter(self._adj[n])

    def number_of_nodes(self):
        return len(self._node)

    def number_of_edges(self):
        return sum(len(a) for a in self._adj.values()) // 2

    def __contains__(self, n):
        return n in self._node

    def __len__(self):
        return len(self._node)

    def subgraph(self, nodes):
        keep = set(nodes) & set(self._node)
        h = Graph()
        for n in keep:
            h.add_node(n, **self._node[n])
        for u in keep:
            for v, attrs in self._adj[u].items():
                if v in keep:
                    h.add_edge(u, v, **attrs)
        return h

    def copy(self):
        return self.subgraph(list(self._node))


def density(G):
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    return 2.0 * G.number_of_edges() / (n * (n - 1))


def number_connected_components(G):
    seen, comps = set(), 0
    for start in G.nodes:
        if start in seen:
            continue
        comps += 1
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            for v in G._adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return comps


def degree_centrality(G):
    n = G.number_of_nodes()
    if n <= 1:
        return {u: 0.0 for u in G.nodes}
    return {u: d / (n - 1) for u, d in G.degree()}


def spring_layout(G, seed=42, k=None, iterations=60, scale=1.0):
    """Fruchterman-Reingold layout (vectorised, adequate for a few hundred nodes)."""
    nodes = list(G.nodes)
    n = len(nodes)
    if n == 0:
        return {}
    ix = {u: i for i, u in enumerate(nodes)}
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * 2 - 1
    A = np.zeros((n, n))
    for u, v in G.edges():
        A[ix[u], ix[v]] = A[ix[v], ix[u]] = 1.0
    k = k or (1.0 / np.sqrt(n))
    t = 0.1 + 0.4 * scale
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, 1e9)
        dist = np.clip(dist, 0.005, None)
        rep = (k ** 2 / dist ** 2)[:, :, None] * delta
        att = (A * dist / k)[:, :, None] * delta
        disp = (rep - att).sum(axis=1)
        length = np.clip(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9, None)
        pos += disp / length * np.minimum(length, t)
        t *= 0.94
    pos -= pos.mean(axis=0)
    span = np.abs(pos).max() or 1.0
    pos = pos / span * scale
    return {u: pos[ix[u]] for u in nodes}
