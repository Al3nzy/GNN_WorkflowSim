"""
Builds production_data.pkl, the training set for the single deployed
LIWSA-GNN instance (train_production.py), as distinct from the held-out
train/test splits used for the scale- and topology-generalisation
studies (build_dataset.py, build_family_datasets.py).

Unlike those two generalisation studies, the production dataset pools
samples from all five workflow families at all four instance sizes,
with nothing held out, since the deployed model is meant to warm-start
LIWSA on any of the twenty benchmark instances at scheduling time, not
to be tested for generalisation to something it has not seen.

Uses the same per-workflow sampling function and sample count per
workflow (SAMPLES_PER_TRAIN_WORKFLOW = 300) as build_dataset.py for
consistency with the rest of the pipeline. This script was not present
in the original repository snapshot; it is added here to make the
Python pipeline runnable end to end. If you already have a
production_data.pkl from your own original run, prefer that one, since
this script's output is only guaranteed to follow the same generation
methodology, not to be byte-identical to it.
"""
import os
import pickle
from training_data import load_workflow, sample_training_instance
from build_dataset import build_dataset, SAMPLES_PER_TRAIN_WORKFLOW

ALL_WORKFLOWS = [
    'Montage_25', 'Montage_50', 'Montage_100', 'Montage_1000',
    'CyberShake_30', 'CyberShake_50', 'CyberShake_100', 'CyberShake_1000',
    'Sipht_30', 'Sipht_60', 'Sipht_100', 'Sipht_1000',
    'Epigenomics_24', 'Epigenomics_46', 'Epigenomics_100', 'Epigenomics_997',
    'Inspiral_30', 'Inspiral_50', 'Inspiral_100', 'Inspiral_1000',
]

if __name__ == '__main__':
    print('Building PRODUCTION set (all 20 workflows, all families, all scales, nothing held out)...')
    production_data = build_dataset(ALL_WORKFLOWS, SAMPLES_PER_TRAIN_WORKFLOW, seed=3)
    print(f'Production set: {len(production_data)} total samples')

    with open('production_data.pkl', 'wb') as f:
        pickle.dump(production_data, f)
    print('\nSaved production_data.pkl')
