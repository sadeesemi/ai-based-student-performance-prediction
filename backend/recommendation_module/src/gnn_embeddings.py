"""Phase 1.5 - Resource mapping with a Graph Neural Network.

A two-layer GraphSAGE encoder (mean aggregator) trained with an unsupervised
link-prediction objective and negative sampling.  Implemented directly in NumPy
so the module trains on any machine with no deep-learning runtime installed.

    h1 = relu([h0 ; A_mean h0] W1)
    z  =      [h1 ; A_mean h1] W2
    loss = -log sigmoid(z_u . z_v)  -  log(1 - sigmoid(z_u . z_n))
"""
import numpy as np
from scipy import sparse

from . import config


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def node_features(G, nodes, random_dims=12, seed=config.RANDOM_SEED):
    """Attribute features + seeded random identity features.

    The graph is largely feature-less on the learner side, so a fixed random
    signature per node is appended (standard practice for GNNs on structural
    graphs) - it lets message passing separate nodes of the same entity type.
    """
    kinds = {k: i for i, k in enumerate(["student", "lesson", "topic", "resource", "tag", "risk", "style"])}
    F = np.zeros((len(nodes), len(kinds) + 6), dtype=np.float64)
    deg = dict(G.degree())
    max_deg = max(deg.values()) or 1
    for i, n in enumerate(nodes):
        d = G.nodes[n]
        F[i, kinds.get(d["kind"], 0)] = 1.0
        F[i, len(kinds)] = np.log1p(deg[n]) / np.log1p(max_deg)
        if d["kind"] == "student":
            F[i, len(kinds) + 1] = d["engagement"] / 100.0
            F[i, len(kinds) + 2] = d["performance"] / 100.0
        elif d["kind"] == "resource":
            F[i, len(kinds) + 3] = min(d["minutes"], 120) / 120.0
            F[i, len(kinds) + 4] = config.DIFFICULTY_SCALE.get(d["level"], 0.5)
            F[i, len(kinds) + 5] = 1.0
    if random_dims:
        rng = np.random.default_rng(seed)
        F = np.hstack([F, rng.normal(0, 0.6, (len(nodes), random_dims))])
    return F


class GraphSAGE:
    def __init__(self, in_dim, dim=config.GNN_DIM, seed=config.RANDOM_SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(1.0 / (2 * in_dim)), (2 * in_dim, dim))
        self.W2 = rng.normal(0, np.sqrt(1.0 / (2 * dim)), (2 * dim, dim))
        self._m = [np.zeros_like(self.W1), np.zeros_like(self.W2)]
        self._v = [np.zeros_like(self.W1), np.zeros_like(self.W2)]
        self._t = 0

    def forward(self, A, H0):
        M1 = A @ H0
        C1 = np.hstack([H0, M1])
        Z1 = C1 @ self.W1
        H1 = np.maximum(Z1, 0)
        M2 = A @ H1
        C2 = np.hstack([H1, M2])
        Z = C2 @ self.W2
        self._cache = (A, H0, C1, Z1, H1, C2)
        return Z

    def backward(self, dZ):
        A, H0, C1, Z1, H1, C2 = self._cache
        gW2 = C2.T @ dZ
        dC2 = dZ @ self.W2.T
        d = H1.shape[1]
        dH1 = dC2[:, :d] + (A.T @ dC2[:, d:])
        dZ1 = dH1 * (Z1 > 0)
        gW1 = C1.T @ dZ1
        return gW1, gW2

    def step(self, grads, lr):
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (W, g) in enumerate(zip([self.W1, self.W2], grads)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * (g ** 2)
            mh = self._m[i] / (1 - b1 ** self._t)
            vh = self._v[i] / (1 - b2 ** self._t)
            W -= lr * mh / (np.sqrt(vh) + eps)


def train(G, epochs=config.GNN_EPOCHS, lr=config.GNN_LR, seed=config.RANDOM_SEED, verbose=True):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    rows, cols = [], []
    for u, v in G.edges():
        rows += [idx[u], idx[v]]
        cols += [idx[v], idx[u]]
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    deg = np.asarray(A.sum(1)).ravel()
    A = sparse.diags(1.0 / np.maximum(deg, 1)) @ A          # mean aggregator

    H0 = node_features(G, nodes)
    edges = np.array([[idx[u], idx[v]] for u, v in G.edges()])
    rng = np.random.default_rng(seed)
    rng.shuffle(edges)
    n_val = max(200, int(0.1 * len(edges)))
    val_edges, train_edges = edges[:n_val], edges[n_val:]

    # type-aware negative sampling: corrupt the tail with a node of the same entity
    # type, otherwise the task collapses to "is this a student?" and learns nothing.
    kinds = np.array([G.nodes[n]["kind"] for n in nodes])
    kind_pool = {k: np.where(kinds == k)[0] for k in set(kinds.tolist())}
    tail_pool = [kind_pool[kinds[j]] for j in range(N)]

    def sample_negatives(pos):
        tails = np.empty(len(pos), dtype=np.int64)
        for i, j in enumerate(pos[:, 1]):
            pool = tail_pool[j]
            tails[i] = pool[rng.integers(0, len(pool))]
        return np.stack([pos[:, 0], tails], axis=1)

    model = GraphSAGE(H0.shape[1], seed=seed)
    history = []
    batch = min(len(train_edges), 8000)
    for ep in range(epochs):
        pos = train_edges[rng.choice(len(train_edges), batch, replace=False)]
        neg = sample_negatives(pos)
        Z = model.forward(A, H0)
        sp = np.sum(Z[pos[:, 0]] * Z[pos[:, 1]], axis=1)
        sn = np.sum(Z[neg[:, 0]] * Z[neg[:, 1]], axis=1)
        loss = float(np.mean(-np.log(_sigmoid(sp) + 1e-9)) + np.mean(-np.log(1 - _sigmoid(sn) + 1e-9)))
        gp = (_sigmoid(sp) - 1.0)[:, None] / batch
        gn = (_sigmoid(sn))[:, None] / batch
        dZ = np.zeros_like(Z)
        np.add.at(dZ, pos[:, 0], gp * Z[pos[:, 1]])
        np.add.at(dZ, pos[:, 1], gp * Z[pos[:, 0]])
        np.add.at(dZ, neg[:, 0], gn * Z[neg[:, 1]])
        np.add.at(dZ, neg[:, 1], gn * Z[neg[:, 0]])
        model.step(model.backward(dZ), lr)
        auc = link_auc(Z, val_edges, sample_negatives(val_edges)[:, 1])
        history.append({"epoch": ep + 1, "loss": round(loss, 4), "val_auc": round(auc, 4)})
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"      GNN epoch {ep+1:>3}/{epochs}  loss={loss:.4f}  val_link_auc={auc:.4f}")

    Z = model.forward(A, H0)
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    emb = Z / np.maximum(norm, 1e-9)
    return {"nodes": nodes, "embeddings": emb, "history": history, "model": model}


def link_auc(Z, val_edges, neg_tails):
    """Held-out link prediction AUC against type-matched corrupted tails."""
    pos = np.sum(Z[val_edges[:, 0]] * Z[val_edges[:, 1]], axis=1)
    neg = np.sum(Z[val_edges[:, 0]] * Z[neg_tails], axis=1)
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())
