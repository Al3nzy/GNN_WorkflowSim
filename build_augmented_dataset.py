import os
import random
import pickle
from training_data import build_graph_features, sample_training_instance, build_vms
from decoder import decode_order, average_bandwidth, decode
from dax_parser import transfer_size_bytes
from synthetic_dag import build_synthetic_batch

FAMILIES = {
    'Montage': ['Montage_25', 'Montage_50', 'Montage_100', 'Montage_1000'],
    'CyberShake': ['CyberShake_30', 'CyberShake_50', 'CyberShake_100', 'CyberShake_1000'],
    'Sipht': ['Sipht_30', 'Sipht_60', 'Sipht_100', 'Sipht_1000'],
    'Epigenomics': ['Epigenomics_24', 'Epigenomics_46', 'Epigenomics_100', 'Epigenomics_997'],
    'Inspiral': ['Inspiral_30', 'Inspiral_50', 'Inspiral_100', 'Inspiral_1000'],
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAX_DIR = os.path.join(SCRIPT_DIR, 'dax')
if not os.path.isdir(DAX_DIR):
    legacy_dir = os.path.join(SCRIPT_DIR, 'config', 'dax')
    if os.path.isdir(legacy_dir):
        DAX_DIR = legacy_dir
SAMPLES_PER_WORKFLOW = 200
N_SYNTHETIC_GRAPHS = 60
SAMPLES_PER_SYNTHETIC_GRAPH = 15


def sample_synthetic_instance(tasks, rng):
    task_order, depth = decode_order(tasks)
    struct_feats = build_graph_features(tasks, task_order, depth)
    vms = build_vms()
    avg_bw = average_bandwidth(vms)
    return sample_training_instance(tasks, task_order, depth, struct_feats, vms, avg_bw, rng)


def build_real_dataset(workflow_names, samples_per_workflow, seed):
    from training_data import load_workflow
    rng = random.Random(seed)
    dataset = []
    for name in workflow_names:
        path = os.path.join(DAX_DIR, f'{name}.xml')
        tasks, task_order, depth, struct_feats, vms, avg_bw = load_workflow(path)
        for _ in range(samples_per_workflow):
            node_features, edges, norm_makespan, norm_cost = sample_training_instance(
                tasks, task_order, depth, struct_feats, vms, avg_bw, rng)
            dataset.append({
                'workflow': name, 'node_features': node_features, 'edges': edges,
                'makespan': norm_makespan, 'cost': norm_cost,
            })
    return dataset


def build_synthetic_dataset(n_graphs, samples_per_graph, seed):
    rng = random.Random(seed)
    graphs = build_synthetic_batch(rng, n_graphs, size_range=(20, 300))
    dataset = []
    for tasks in graphs:
        for _ in range(samples_per_graph):
            node_features, edges, norm_makespan, norm_cost = sample_synthetic_instance(tasks, rng)
            dataset.append({
                'workflow': 'synthetic', 'node_features': node_features, 'edges': edges,
                'makespan': norm_makespan, 'cost': norm_cost,
            })
    return dataset


if __name__ == '__main__':
    import sys
    held_out_family = sys.argv[1]
    train_workflows = [w for fam, wfs in FAMILIES.items() if fam != held_out_family for w in wfs]

    print(f'Building AUGMENTED training set (held out: {held_out_family})...')
    real_data = build_real_dataset(train_workflows, SAMPLES_PER_WORKFLOW, seed=10)
    print(f'  Real workflow samples: {len(real_data)}')

    synthetic_data = build_synthetic_dataset(N_SYNTHETIC_GRAPHS, SAMPLES_PER_SYNTHETIC_GRAPH, seed=99)
    print(f'  Synthetic DAG samples: {len(synthetic_data)} (from {N_SYNTHETIC_GRAPHS} distinct random graphs)')

    combined = real_data + synthetic_data
    random.Random(5).shuffle(combined)

    with open(f'augmented_train_{held_out_family}.pkl', 'wb') as f:
        pickle.dump(combined, f)
    print(f'  Total augmented training set: {len(combined)} samples saved')
