import pickle
import random
import torch
from train_lib import train_gnn_one_seed, evaluate_gnn
from model import WorkflowGNN
import torch.nn as nn

with open('production_data.pkl', 'rb') as f:
    data = pickle.load(f)

rng = random.Random(42)
rng.shuffle(data)
split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]
print(f'Production training: {len(train_data)} train, {len(val_data)} validation samples')

result = train_gnn_one_seed(lambda: WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3, dropout=0.1),
                             train_data, val_data, seed=1, n_epochs=15, verbose=True)

print(f'\nFinal production model:')
print(f'  train MSE={result["train_mse"]:.4f}')
print(f'  val MSE={result["test_mse"]:.4f}  MAE(makespan)={result["test_mae_makespan"]:.4f}  MAE(cost)={result["test_mae_cost"]:.4f}')

torch.save(result['model'].state_dict(), 'production_gnn.pt')
print('\nSaved to production_gnn.pt')
