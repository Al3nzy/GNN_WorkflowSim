import sys
import pickle
import os
from train_lib import train_gnn_one_seed
from model import WorkflowGNN
from train_baseline import FlatBaseline

seed = int(sys.argv[1])
N_EPOCHS = 15  # reduced further after seed 2 timed out at 25 epochs

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

results_path = 'multiseed_checkpoint.pkl'
if os.path.exists(results_path):
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
else:
    results = {'GNN': [], 'FlatBaseline': []}

# Skip if this seed was already completed (resumable)
if any(r['seed'] == seed for r in results['GNN']):
    print(f'Seed {seed} already completed, skipping.')
    sys.exit(0)

print(f'=== Seed {seed}: GNN ===')
r = train_gnn_one_seed(lambda: WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3),
                        train_data, test_data, seed, n_epochs=N_EPOCHS, verbose=False)
print(f'  train_MSE={r["train_mse"]:.4f}  test_MSE={r["test_mse"]:.4f}  '
      f'test_MAE(makespan)={r["test_mae_makespan"]:.4f}  test_MAE(cost)={r["test_mae_cost"]:.4f}')
results['GNN'].append({k: v for k, v in r.items() if k != 'model'})

print(f'=== Seed {seed}: FlatBaseline ===')
r2 = train_gnn_one_seed(lambda: FlatBaseline(in_dim=6, hidden_dim=32),
                         train_data, test_data, seed, n_epochs=N_EPOCHS, verbose=False)
print(f'  train_MSE={r2["train_mse"]:.4f}  test_MSE={r2["test_mse"]:.4f}  '
      f'test_MAE(makespan)={r2["test_mae_makespan"]:.4f}  test_MAE(cost)={r2["test_mae_cost"]:.4f}')
results['FlatBaseline'].append({k: v for k, v in r2.items() if k != 'model'})

with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f'\nCheckpoint saved. Seeds completed so far: {[r["seed"] for r in results["GNN"]]}')
