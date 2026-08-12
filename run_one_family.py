import sys
import pickle
import os
from train_lib import train_gnn_one_seed
from model import WorkflowGNN
from train_baseline import FlatBaseline

family = sys.argv[1]
N_EPOCHS = 15
SEED = 1

with open(f'family_train_{family}.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(f'family_test_{family}.pkl', 'rb') as f:
    test_data = pickle.load(f)

results_path = 'family_holdout_checkpoint.pkl'
if os.path.exists(results_path):
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
else:
    results = {}

if family in results:
    print(f'{family} already completed, skipping.')
    sys.exit(0)

print(f'=== Held-out family: {family} ({len(train_data)} train, {len(test_data)} test) ===')

gnn_result = train_gnn_one_seed(lambda: WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3),
                                 train_data, test_data, SEED, n_epochs=N_EPOCHS, verbose=False)
print(f'  GNN:          test_MSE={gnn_result["test_mse"]:.4f}  '
      f'MAE(makespan)={gnn_result["test_mae_makespan"]:.4f}  MAE(cost)={gnn_result["test_mae_cost"]:.4f}')

baseline_result = train_gnn_one_seed(lambda: FlatBaseline(in_dim=6, hidden_dim=32),
                                      train_data, test_data, SEED, n_epochs=N_EPOCHS, verbose=False)
print(f'  FlatBaseline: test_MSE={baseline_result["test_mse"]:.4f}  '
      f'MAE(makespan)={baseline_result["test_mae_makespan"]:.4f}  MAE(cost)={baseline_result["test_mae_cost"]:.4f}')

ratio = baseline_result['test_mse'] / gnn_result['test_mse']
print(f'  GNN advantage: {ratio:.2f}x lower test MSE than flat baseline')

results[family] = {
    'gnn': {k: v for k, v in gnn_result.items() if k != 'model'},
    'baseline': {k: v for k, v in baseline_result.items() if k != 'model'},
    'advantage_ratio': ratio,
}
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f'\nCheckpoint saved. Families completed so far: {list(results.keys())}')
