"""
A small, dependency-free graph neural network for predicting normalized
makespan and cost from a workflow DAG plus a candidate VM assignment.

Message passing is direction-aware, which matters for DAGs: a task's
achievable finish time depends on what its ancestors did, and a task's
downstream importance depends on what its descendants need. Each layer
therefore aggregates separately from parents and from children (mean
aggregation via row-normalised sparse adjacency matmuls), rather than
treating the DAG as an undirected graph the way a naive GCN would.

No torch_geometric dependency: each graph's two directional adjacency
matrices are built as sparse tensors, and aggregation is a single sparse
matmul per direction per layer, which is both correct and fast enough
for graphs up to ~1000 nodes without needing torch_scatter.
"""
import torch
import torch.nn as nn


def build_adjacency(edges, n_nodes):
    """Returns (parent_to_child, child_to_parent) row-normalised sparse
    adjacency matrices. parent_to_child[c, p] = 1/fanin(c): aggregating
    with this matrix gives, for each node, the mean of its parents'
    embeddings. child_to_parent is the transpose-normalised counterpart,
    giving each node the mean of its children's embeddings."""
    if not edges:
        z = torch.sparse_coo_tensor(size=(n_nodes, n_nodes))
        return z, z

    parents = [e[0] for e in edges]
    children = [e[1] for e in edges]

    fanin = [0] * n_nodes
    fanout = [0] * n_nodes
    for p, c in edges:
        fanin[c] += 1
        fanout[p] += 1

    pc_idx = torch.tensor([children, parents], dtype=torch.long)
    pc_val = torch.tensor([1.0 / fanin[c] for c in children], dtype=torch.float32)
    parent_to_child = torch.sparse_coo_tensor(pc_idx, pc_val, (n_nodes, n_nodes)).coalesce()

    cp_idx = torch.tensor([parents, children], dtype=torch.long)
    cp_val = torch.tensor([1.0 / fanout[p] for p in parents], dtype=torch.float32)
    child_to_parent = torch.sparse_coo_tensor(cp_idx, cp_val, (n_nodes, n_nodes)).coalesce()

    return parent_to_child, child_to_parent


class DAGMessagePassingLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.parent_lin = nn.Linear(in_dim, out_dim)
        self.child_lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, parent_to_child, child_to_parent):
        from_parents = torch.sparse.mm(parent_to_child, h)
        from_children = torch.sparse.mm(child_to_parent, h)
        out = self.self_lin(h) + self.parent_lin(from_parents) + self.child_lin(from_children)
        return self.dropout(self.act(out))


class WorkflowGNN(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=32, n_layers=3, dropout=0.1):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            DAGMessagePassingLayer(dims[i], dims[i + 1], dropout=dropout) for i in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [makespan, cost], both normalised
        )

    def forward(self, node_features, edges):
        n = node_features.shape[0]
        parent_to_child, child_to_parent = build_adjacency(edges, n)
        h = node_features
        for layer in self.layers:
            h = layer(h, parent_to_child, child_to_parent)
        graph_embedding = h.mean(dim=0)  # global mean pooling
        return self.head(graph_embedding)
