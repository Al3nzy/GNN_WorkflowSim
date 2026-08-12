"""
Harder generalization test than build_dataset.py: instead of holding out
just the largest scale of every family, this holds out one ENTIRE family
(all four of its scale points) and trains on the other four families
completely. This tests topology generalization: Inspiral's DAG shape
(shallow, wide-fan-in at each level) is structurally different from
Sipht's (deep sequential chains), so a model that only learned
"Montage/CyberShake/Sipht/Epigenomics-shaped graphs" has to extrapolate
to a genuinely different structure it has never encountered, not just a
bigger version of a familiar shape.

Run once per held-out family (5 runs total) to see whether some
topologies are harder to generalize to than others, which is itself a
useful, honestly-reported finding for the paper.
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

FAMILIES = {
    'Montage': ['Montage_25', 'Montage_50', 'Montage_100', 'Montage_1000'],
    'CyberShake': ['CyberShake_30', 'CyberShake_50', 'CyberShake_100', 'CyberShake_1000'],
    'Sipht': ['Sipht_30', 'Sipht_60', 'Sipht_100', 'Sipht_1000'],
    'Epigenomics': ['Epigenomics_24', 'Epigenomics_46', 'Epigenomics_100', 'Epigenomics_997'],
    'Inspiral': ['Inspiral_30', 'Inspiral_50', 'Inspiral_100', 'Inspiral_1000'],
}

SAMPLES_PER_WORKFLOW = 200  # slightly fewer than the scale-generalization run,
                             # since this run repeats 5x (once per held-out family)


def build_dataset(workflow_names, samples_per_workflow, seed):
    rng = random.Random(seed)
    dataset = []
    for name in workflow_names:
        path = os.path.join(DAX_DIR, f'{name}.xml')
        tasks, task_order, depth, struct_feats, vms, avg_bw = load_workflow(path)
        for _ in range(samples_per_workflow):
            node_features, edges, norm_makespan, norm_cost = sample_training_instance(
                tasks, task_order, depth, struct_feats, vms, avg_bw, rng)
            dataset.append({
                'workflow': name, 'n_tasks': len(tasks),
                'node_features': node_features, 'edges': edges,
                'makespan': norm_makespan, 'cost': norm_cost,
            })
    return dataset


if __name__ == '__main__':
    for held_out_family in FAMILIES:
        train_workflows = [w for fam, wfs in FAMILIES.items() if fam != held_out_family for w in wfs]
        test_workflows = FAMILIES[held_out_family]

        print(f'Held-out family: {held_out_family}')
        print(f'  Train: {len(train_workflows)} workflows from {len(FAMILIES)-1} other families')
        print(f'  Test:  {test_workflows}')

        train_data = build_dataset(train_workflows, SAMPLES_PER_WORKFLOW, seed=10)
        test_data = build_dataset(test_workflows, SAMPLES_PER_WORKFLOW, seed=20)

        with open(f'family_train_{held_out_family}.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open(f'family_test_{held_out_family}.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        print(f'  Saved: {len(train_data)} train, {len(test_data)} test samples\n')
