"""
Exports every learnable parameter of the trained WorkflowGNN to plain JSON
numbers, so the exact same forward pass can be reimplemented in pure Java
with no ML runtime dependency, matching LIWSA-ML's own "zero external
libraries" design. This is deliberately chosen over ONNX export: this
model uses torch.sparse operations for DAG-directional message passing,
and ONNX's sparse-op coverage is historically incomplete/unreliable, so a
direct, hand-verified weight port is the more robust path here, not a
compromise.
"""
import json
import torch
from model import WorkflowGNN

model = WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3, dropout=0.1)
model.load_state_dict(torch.load('production_gnn.pt'))
model.eval()

weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().numpy().tolist()

with open('gnn_weights.json', 'w') as f:
    json.dump(weights, f)

print(f"Exported {len(weights)} parameter tensors to gnn_weights.json")
for name, w in weights.items():
    shape = _shape = None
    import numpy as np
    arr = np.array(w)
    print(f"  {name}: shape={arr.shape}")
