import sys
import pickle
from train_lib import train_gnn_one_seed
from model import WorkflowGNN

family = sys.argv[1]
N_EPOCHS = 15
SEED = 1

with open(f'augmented_train_{family}.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(f'family_test_{family}.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f'=== AUGMENTED training, held-out family: {family} ===')
print(f'  Train: {len(train_data)} (real + synthetic), Test: {len(test_data)} (real, held out)')

result = train_gnn_one_seed(lambda: WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3, dropout=0.1),
                             train_data, test_data, SEED, n_epochs=N_EPOCHS, verbose=False)
print(f'  GNN (augmented): test_MSE={result["test_mse"]:.4f}  '
      f'MAE(makespan)={result["test_mae_makespan"]:.4f}  MAE(cost)={result["test_mae_cost"]:.4f}')

with open('family_holdout_checkpoint.pkl', 'rb') as f:
    prev = pickle.load(f)
old_gnn_mse = prev[family]['gnn']['test_mse']
old_baseline_mse = prev[family]['baseline']['test_mse']
print(f'\n  Previous (real-only) GNN test_MSE:  {old_gnn_mse:.4f}')
print(f'  Previous FlatBaseline test_MSE:      {old_baseline_mse:.4f}')
print(f'  New (augmented) GNN test_MSE:        {result["test_mse"]:.4f}')
improvement = old_gnn_mse / result['test_mse']
vs_baseline = old_baseline_mse / result['test_mse']
print(f'\n  Augmentation improved GNN by: {improvement:.2f}x (lower MSE is better)')
print(f'  Augmented GNN vs FlatBaseline: {vs_baseline:.2f}x ({"GNN now wins" if vs_baseline > 1 else "baseline still wins"})')
