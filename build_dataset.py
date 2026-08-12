"""
Builds the train/test split for the GNN generalization experiment.

TRAIN: 16 workflow instances (the 25/30/24/29/30-, 50/46/47/58/60-, and
100-task instance of every family) -- every family and DAG topology is
represented, but only at small-to-medium scale.

TEST (held out entirely, never seen in training): the four 1000-task
instances, one per family (Montage_1000, CyberShake_1000, Sipht_1000,
Epigenomics_1000-equivalent = Epigenomics_997, Inspiral_1000). This
tests whether the model generalises to scales far beyond anything it
trained on, the harder and more meaningful of the two plausible
generalisation tests (the other being held-out families, which a
follow-up experiment can add).
"""
import os
import random
import pickle
from training_data import load_workflow, sample_training_instance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAX_DIR = os.path.join(SCRIPT_DIR, 'dax')
if not os.path.isdir(DAX_DIR):
    legacy_dir = os.path.join(SCRIPT_DIR, 'config', 'dax')
    if os.path.isdir(legacy_dir):
        DAX_DIR = legacy_dir

TRAIN_WORKFLOWS = [
    'Montage_25', 'Montage_50', 'Montage_100',
    'CyberShake_30', 'CyberShake_50', 'CyberShake_100',
    'Sipht_30', 'Sipht_60', 'Sipht_100',
    'Epigenomics_24', 'Epigenomics_46', 'Epigenomics_100',
    'Inspiral_30', 'Inspiral_50', 'Inspiral_100',
]
TEST_WORKFLOWS = [
    'Montage_1000', 'CyberShake_1000', 'Sipht_1000', 'Epigenomics_997', 'Inspiral_1000',
]

SAMPLES_PER_TRAIN_WORKFLOW = 300
SAMPLES_PER_TEST_WORKFLOW = 60


def build_dataset(workflow_names, samples_per_workflow, seed):
    rng = random.Random(seed)
    dataset = []
    for name in workflow_names:
        path = os.path.join(DAX_DIR, f'{name}.xml')
        tasks, task_order, depth, struct_feats, vms, avg_bw = load_workflow(path)
        n_tasks = len(tasks)
        for _ in range(samples_per_workflow):
            node_features, edges, norm_makespan, norm_cost = sample_training_instance(
                tasks, task_order, depth, struct_feats, vms, avg_bw, rng)
            dataset.append({
                'workflow': name,
                'n_tasks': n_tasks,
                'node_features': node_features,
                'edges': edges,
                'makespan': norm_makespan,
                'cost': norm_cost,
            })
        print(f'  {name}: {n_tasks} tasks, {samples_per_workflow} samples generated')
    return dataset


if __name__ == '__main__':
    print('Building TRAIN set (16 workflows, small-to-medium scale)...')
    train_data = build_dataset(TRAIN_WORKFLOWS, SAMPLES_PER_TRAIN_WORKFLOW, seed=1)
    print(f'Train set: {len(train_data)} total samples\n')

    print('Building TEST set (4 held-out 1000-task workflows, never seen above)...')
    test_data = build_dataset(TEST_WORKFLOWS, SAMPLES_PER_TEST_WORKFLOW, seed=2)
    print(f'Test set: {len(test_data)} total samples')

    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    print('\nSaved train_data.pkl and test_data.pkl')
