"""
Extends family-holdout generalization to 5 seeds per held-out family
(matching scale-generalization's rigor), at 40 epochs.

Starts from your existing family_holdout_checkpoint_40ep.pkl (which has
seed 1 for each family, from the run you already did) and adds seeds
2-5. Seed 1 is NOT redone -- its result is carried over as-is.

------------------------------------------------------------------
BEFORE RUNNING, this needs to be in the same directory as your other
paper-2 files (model.py, decoder.py, dax_parser.py, training_data.py,
train_lib.py, build_family_datasets.py, train_baseline.py, config/dax/),
and ideally your existing family_holdout_checkpoint_40ep.pkl and the
family_train_*.pkl / family_test_*.pkl files from the last run (if
those aren't present, this script rebuilds them the same way
run_full_sweep.py did -- same seeds, same function calls, so the data
will be identical either way).

Run:  python run_family_multiseed.py

Resumable: saves after every (family, seed) pair, safe to stop and
restart across multiple Colab sessions. At ~600s/run (40 epochs, what
you measured), 5 families x 2 models x 4 new seeds is roughly 6.5-7
hours total -- expect to run this across more than one sitting.
------------------------------------------------------------------

Output: family_holdout_checkpoint_40ep_multiseed.pkl
  {family: {'gnn': [5 per-seed result dicts], 'baseline': [5 per-seed result dicts]}}
"""
import os
import pickle
import time
from pathlib import Path

N_EPOCHS = 40
SEEDS = [1, 2, 3, 4, 5]
BASE_DIR = Path(__file__).resolve().parent

from train_lib import train_gnn_one_seed
from model import WorkflowGNN
from train_baseline import FlatBaseline
from build_family_datasets import FAMILIES, SAMPLES_PER_WORKFLOW
from build_family_datasets import build_dataset as build_family_data

ckpt_path = BASE_DIR / 'family_holdout_checkpoint_40ep_multiseed.pkl'


def normalize_seed_results(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


if ckpt_path.exists():
    with ckpt_path.open('rb') as f:
        results = pickle.load(f)
    print('Resuming from existing multiseed checkpoint.')
else:
    results = {}
    old_path = BASE_DIR / 'family_holdout_checkpoint_40ep.pkl'
    if old_path.exists():
        with old_path.open('rb') as f:
            old = pickle.load(f)
    else:
        old = {}
    for family in FAMILIES:
        old_family = old.get(family, {})
        results[family] = {
            'gnn': normalize_seed_results(old_family.get('gnn')),
            'baseline': normalize_seed_results(old_family.get('baseline')),
        }
    with ckpt_path.open('wb') as f:
        pickle.dump(results, f)
    print('Seeded from existing n=1 checkpoint (or starting fresh where absent).')

print('\nStarting point:')
for family in FAMILIES:
    done = sorted(r['seed'] for r in results[family]['gnn'])
    print(f'  {family}: seeds already done = {done}')
print()

for family in FAMILIES:
    train_path = BASE_DIR / f'family_train_{family}.pkl'
    test_path = BASE_DIR / f'family_test_{family}.pkl'
    if not (train_path.exists() and test_path.exists()):
        print(f'Building {family} data...')
        train_workflows = [w for fam, wfs in FAMILIES.items() if fam != family for w in wfs]
        test_workflows = FAMILIES[family]
        tr = build_family_data(train_workflows, SAMPLES_PER_WORKFLOW, seed=10)
        te = build_family_data(test_workflows, SAMPLES_PER_WORKFLOW, seed=20)
        with train_path.open('wb') as f:
            pickle.dump(tr, f)
        with test_path.open('wb') as f:
            pickle.dump(te, f)

    with train_path.open('rb') as f:
        tr = pickle.load(f)
    with test_path.open('rb') as f:
        te = pickle.load(f)

    done_seeds = {r.get('seed') for r in results[family]['gnn'] if isinstance(r, dict) and 'seed' in r}
    for seed in SEEDS:
        if seed in done_seeds:
            print(f'{family} seed {seed}: already done, skipping.')
            continue
        t0 = time.time()
        gnn_r = train_gnn_one_seed(lambda: WorkflowGNN(in_dim=6, hidden_dim=32, n_layers=3),
                                    tr, te, seed=seed, n_epochs=N_EPOCHS, verbose=False)
        base_r = train_gnn_one_seed(lambda: FlatBaseline(in_dim=6, hidden_dim=32),
                                     tr, te, seed=seed, n_epochs=N_EPOCHS, verbose=False)
        results[family]['gnn'].append({k: v for k, v in gnn_r.items() if k != 'model'})
        results[family]['baseline'].append({k: v for k, v in base_r.items() if k != 'model'})
        with ckpt_path.open('wb') as f:
            pickle.dump(results, f)
        ratio = base_r['test_mse'] / gnn_r['test_mse']
        winner = 'GNN' if ratio > 1 else 'Baseline'
        print(f'{family} seed {seed}: done in {time.time()-t0:.0f}s -- '
              f'GNN={gnn_r["test_mse"]:.4f}  Baseline={base_r["test_mse"]:.4f}  '
              f'({winner} wins, {max(ratio, 1/ratio):.2f}x)')

print('\nDone. Per-family seed counts:')
for family in FAMILIES:
    print(f'  {family}: {len(results[family]["gnn"])} seeds')
print(f'\nSaved to {ckpt_path}')
