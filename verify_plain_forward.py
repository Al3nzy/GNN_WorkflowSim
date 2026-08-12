"""
Two things:
1. Convert gnn_weights.json into a simple flat text format Java can parse
   with basic String.split(), no JSON library needed.
2. A PyTorch-free reference forward pass, using only plain lists and loops,
   mirroring exactly what the Java implementation will do. This is the
   critical validation step: if this plain-Python version matches
   PyTorch's real output exactly, then a faithful, mechanical translation
   into Java (same loops, same arithmetic, same order of operations) can
   be trusted without needing a Java compiler in this sandbox to check it.
"""
import json
import math

with open('gnn_weights.json') as f:
    W = json.load(f)

# ---- 1. Flat text export for Java ----
with open('gnn_weights.txt', 'w') as f:
    for name, tensor in W.items():
        flat = []
        def flatten(x):
            if isinstance(x, list):
                for v in x:
                    flatten(v)
            else:
                flat.append(x)
        flatten(tensor)
        f.write(f"{name} {len(flat)} {' '.join(str(v) for v in flat)}\n")
print("Wrote gnn_weights.txt")


# ---- 2. Plain-Python (no torch) reference forward pass ----
def linear(x, weight, bias):
    """x: list[float] (in_dim). weight: list[list[float]] (out_dim x in_dim).
    bias: list[float] (out_dim). Returns list[float] (out_dim)."""
    out_dim = len(weight)
    in_dim = len(x)
    out = [0.0] * out_dim
    for o in range(out_dim):
        s = bias[o]
        row = weight[o]
        for i in range(in_dim):
            s += row[i] * x[i]
        out[o] = s
    return out


def relu(x):
    return [max(0.0, v) for v in x]


def dag_layer(h, edges, n_nodes, self_w, self_b, parent_w, parent_b, child_w, child_b):
    """h: list of node feature vectors (n_nodes x in_dim).
    Returns new list of node feature vectors (n_nodes x out_dim)."""
    fanin = [0] * n_nodes
    fanout = [0] * n_nodes
    for p, c in edges:
        fanin[c] += 1
        fanout[p] += 1

    from_parents = [[0.0] * len(h[0]) for _ in range(n_nodes)]
    from_children = [[0.0] * len(h[0]) for _ in range(n_nodes)]
    for p, c in edges:
        w_pc = 1.0 / fanin[c]
        w_cp = 1.0 / fanout[p]
        for k in range(len(h[0])):
            from_parents[c][k] += h[p][k] * w_pc
            from_children[p][k] += h[c][k] * w_cp

    new_h = []
    for i in range(n_nodes):
        a = linear(h[i], self_w, self_b)
        b = linear(from_parents[i], parent_w, parent_b)
        c = linear(from_children[i], child_w, child_b)
        combined = [a[k] + b[k] + c[k] for k in range(len(a))]
        new_h.append(relu(combined))
    return new_h


def gnn_forward_plain(node_features, edges, weights):
    n_nodes = len(node_features)
    h = node_features
    for layer_idx in range(3):
        prefix = f'layers.{layer_idx}'
        h = dag_layer(
            h, edges, n_nodes,
            weights[f'{prefix}.self_lin.weight'], weights[f'{prefix}.self_lin.bias'],
            weights[f'{prefix}.parent_lin.weight'], weights[f'{prefix}.parent_lin.bias'],
            weights[f'{prefix}.child_lin.weight'], weights[f'{prefix}.child_lin.bias'],
        )
    # global mean pooling
    hidden_dim = len(h[0])
    graph_embedding = [sum(h[i][k] for i in range(n_nodes)) / n_nodes for k in range(hidden_dim)]
    # head: Linear -> ReLU -> (dropout skipped at inference) -> Linear
    x = linear(graph_embedding, weights['head.0.weight'], weights['head.0.bias'])
    x = relu(x)
    out = linear(x, weights['head.3.weight'], weights['head.3.bias'])
    return out


if __name__ == '__main__':
    import pickle
    import torch
    from model import WorkflowGNN

    with open('production_data.pkl', 'rb') as f:
        data = pickle.load(f)

    model = WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3, dropout=0.1)
    model.load_state_dict(torch.load('production_gnn.pt'))
    model.eval()

    print("Validating plain-Python reference against real PyTorch model on 5 samples:\n")
    max_diff = 0.0
    for i in range(5):
        sample = data[i]
        x_torch = torch.tensor(sample['node_features'], dtype=torch.float32)
        with torch.no_grad():
            torch_out = model(x_torch, sample['edges']).tolist()

        plain_out = gnn_forward_plain(sample['node_features'], sample['edges'], W)

        diff = max(abs(a - b) for a, b in zip(torch_out, plain_out))
        max_diff = max(max_diff, diff)
        print(f"  Sample {i} ({sample['workflow']}, {len(sample['node_features'])} tasks):")
        print(f"    PyTorch: {[round(v,6) for v in torch_out]}")
        print(f"    Plain:   {[round(v,6) for v in plain_out]}")
        print(f"    max abs diff: {diff:.8f}")

    print(f"\nOverall max abs diff across all 5 samples: {max_diff:.8f}")
    print("PASS" if max_diff < 1e-4 else "FAIL - implementations diverge")
