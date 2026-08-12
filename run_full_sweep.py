"""
Full 40-epoch reproduction sweep for LIWSA-GNN (paper 2), matching the
epoch count stated in methodology_section.tex. The original checkpoints
you have (multiseed_checkpoint.pkl, family_holdout_checkpoint.pkl) were
produced at 15 epochs -- this script reruns both experiments at 40 so
the paper's stated methodology and the actual results agree.

Nothing about the model, data pipeline, or training procedure is changed
from your original files -- this script only calls your existing
build_dataset() / build_family_datasets-style build_dataset() and
train_gnn_one_seed() functions with n_epochs=40 instead of 15, and adds
a resumable checkpoint loop around them so a Colab disconnect doesn't
lose completed work.

------------------------------------------------------------------
BEFORE RUNNING, put these in the same directory as this script:
  config/dax/*.xml   -- the 20 Pegasus DAX files (from your
                         WorkflowSim_LocustModeling repo's config/dax/)
  model.py, decoder.py, dax_parser.py, training_data.py, train_lib.py,
  build_dataset.py, build_family_datasets.py   -- unchanged, from your
                                                    paper-2 package
  train_baseline.py  -- your original FlatBaseline file

Install once:  pip install torch numpy scipy

Run:           python run_full_sweep.py

Safe to stop and restart (Ctrl+C, or a Colab timeout) -- it saves after
every seed/family and skips whatever's already done.
------------------------------------------------------------------

Output (new files, your original 15-epoch checkpoints are untouched):
  multiseed_checkpoint_40ep.pkl
  family_holdout_checkpoint_40ep.pkl
"""
import os
import pickle
import time

N_EPOCHS = int(os.environ.get('SWEEP_EPOCHS', 40))

print(f'=== LIWSA-GNN full sweep, N_EPOCHS={N_EPOCHS} ===\n')

# ------------------------------------------------------------------
# Part 1: scale-generalization (5 seeds x {GNN, FlatBaseline})
# ------------------------------------------------------------------
if not (os.path.exists('train_data.pkl') and os.path.exists('test_data.pkl')):
    print('[1/4] Building scale-generalization train/test data from DAX files...')
    from build_dataset import (build_dataset as build_scale_data, TRAIN_WORKFLOWS,
                                TEST_WORKFLOWS, SAMPLES_PER_TRAIN_WORKFLOW,
                                SAMPLES_PER_TEST_WORKFLOW)
    _train = build_scale_data(TRAIN_WORKFLOWS, SAMPLES_PER_TRAIN_WORKFLOW, seed=1)
    _test = build_scale_data(TEST_WORKFLOWS, SAMPLES_PER_TEST_WORKFLOW, seed=2)
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(_train, f)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(_test, f)
    print(f'  Built {len(_train)} train, {len(_test)} test samples.\n')
else:
    print('[1/4] train_data.pkl / test_data.pkl already present, skipping build.\n')

from train_lib import train_gnn_one_seed
from model import WorkflowGNN
from train_baseline import FlatBaseline

with open('train_data.pkl', 'rb') as f:
    scale_train = pickle.load(f)
with open('test_data.pkl', 'rb') as f:
    scale_test = pickle.load(f)

ckpt_path = 'multiseed_checkpoint_40ep.pkl'
results = pickle.load(open(ckpt_path, 'rb')) if os.path.exists(ckpt_path) else {'GNN': [], 'FlatBaseline': []}

print('[2/4] Scale-generalization sweep (5 seeds):')
for seed in [1, 2, 3, 4, 5]:
    if any(r['seed'] == seed for r in results['GNN']):
        print(f'  seed {seed}: already done, skipping.')
        continue
    t0 = time.time()
    r = train_gnn_one_seed(lambda: WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3),
                            scale_train, scale_test, seed, n_epochs=N_EPOCHS, verbose=False)
    results['GNN'].append({k: v for k, v in r.items() if k != 'model'})
    r2 = train_gnn_one_seed(lambda: FlatBaseline(in_dim=6, hidden_dim=32),
                             scale_train, scale_test, seed, n_epochs=N_EPOCHS, verbose=False)
    results['FlatBaseline'].append({k: v for k, v in r2.items() if k != 'model'})
    with open(ckpt_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'  seed {seed}: done in {time.time()-t0:.0f}s -- '
          f'GNN test_MSE={r["test_mse"]:.4f}  Baseline test_MSE={r2["test_mse"]:.4f}  '
          f'ratio={r2["test_mse"]/r["test_mse"]:.2f}x')
print()

# ------------------------------------------------------------------
# Part 2: family-generalization (5 held-out families x {GNN, FlatBaseline})
# ------------------------------------------------------------------
from build_family_datasets import FAMILIES, SAMPLES_PER_WORKFLOW
from build_family_datasets import build_dataset as build_family_data

print('[3/4] Building family-holdout data (skips families already built)...')
for held_out in FAMILIES:
    if os.path.exists(f'family_train_{held_out}.pkl') and os.path.exists(f'family_test_{held_out}.pkl'):
        print(f'  {held_out}: already present, skipping.')
        continue
    train_workflows = [w for fam, wfs in FAMILIES.items() if fam != held_out for w in wfs]
    test_workflows = FAMILIES[held_out]
    tr = build_family_data(train_workflows, SAMPLES_PER_WORKFLOW, seed=10)
    te = build_family_data(test_workflows, SAMPLES_PER_WORKFLOW, seed=20)
    with open(f'family_train_{held_out}.pkl', 'wb') as f:
        pickle.dump(tr, f)
    with open(f'family_test_{held_out}.pkl', 'wb') as f:
        pickle.dump(te, f)
    print(f'  {held_out}: built {len(tr)} train, {len(te)} test samples.')
print()

fam_ckpt_path = 'family_holdout_checkpoint_40ep.pkl'
fam_results = pickle.load(open(fam_ckpt_path, 'rb')) if os.path.exists(fam_ckpt_path) else {}

print('[4/4] Family-generalization sweep (5 held-out families):')
for family in FAMILIES:
    if family in fam_results:
        print(f'  {family}: already done, skipping.')
        continue
    with open(f'family_train_{family}.pkl', 'rb') as f:
        tr = pickle.load(f)
    with open(f'family_test_{family}.pkl', 'rb') as f:
        te = pickle.load(f)
    t0 = time.time()
    gnn_r = train_gnn_one_seed(lambda: WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3),
                                tr, te, seed=1, n_epochs=N_EPOCHS, verbose=False)
    base_r = train_gnn_one_seed(lambda: FlatBaseline(in_dim=6, hidden_dim=32),
                                 tr, te, seed=1, n_epochs=N_EPOCHS, verbose=False)
    ratio = base_r['test_mse'] / gnn_r['test_mse']
    fam_results[family] = {
        'gnn': {k: v for k, v in gnn_r.items() if k != 'model'},
        'baseline': {k: v for k, v in base_r.items() if k != 'model'},
        'advantage_ratio': ratio,
    }
    with open(fam_ckpt_path, 'wb') as f:
        pickle.dump(fam_results, f)
    tag = 'GNN wins' if ratio > 1 else 'baseline wins'
    print(f'  {family}: done in {time.time()-t0:.0f}s -- '
          f'GNN test_MSE={gnn_r["test_mse"]:.4f}  Baseline test_MSE={base_r["test_mse"]:.4f}  '
          f'({tag}, {max(ratio, 1/ratio):.2f}x)')

print('\nDone. Results in multiseed_checkpoint_40ep.pkl and family_holdout_checkpoint_40ep.pkl')
