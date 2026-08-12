"""
Multi-workflow training data generation for the GNN predictor.

Key design decision, worth stating explicitly: makespan and cost have
wildly different absolute scales across workflows (Montage_50's makespan
is ~50-400s; Epigenomics_997's can exceed 1,000,000s). Training a single
regression head on raw values would let the largest-scale workflows
dominate the loss entirely. Both targets are therefore normalized
per-workflow, relative to that workflow's own sequential execution time
(sum of task lengths / fastest available MIPS), the same quantity the
paper's speedup metric already uses. This keeps the prediction target
scale-invariant across workflows, which is exactly what's needed for a
model meant to generalize across previously-unseen DAG structures.
"""
import random
from dax_parser import parse_dax, transfer_size_bytes
from decoder import VM, decode, average_bandwidth, decode_order

VM_TYPES = [(250.0, 0.15), (500.0, 0.30), (1000.0, 0.60), (2000.0, 0.90)]


def build_vms():
    return [VM(i, mips, 160.0, cost) for i, (mips, cost) in enumerate(VM_TYPES * 4)]


def sequential_makespan(tasks, fastest_mips):
    return sum(t.length for t in tasks) / fastest_mips


def build_graph_features(tasks, task_order, depth):
    """Structural, per-workflow-relative features for each task, independent
    of any specific VM assignment. Computed once per workflow."""
    max_len = max(t.length for t in tasks)
    max_depth = max(depth.values()) if depth else 1
    max_fanout = max(len(t.children) for t in tasks) or 1
    max_fanin = max(len(t.parents) for t in tasks) or 1

    feats = {}
    for t in tasks:
        feats[t.id] = [
            t.length / max_len,
            depth[t.id] / max(max_depth, 1),
            len(t.children) / max_fanout,
            len(t.parents) / max_fanin,
        ]
    return feats


def sample_training_instance(tasks, task_order, depth, struct_feats, vms, avg_bw, rng):
    """One random genotype -> (per-node features incl. VM assignment,
    edge list, normalized makespan, normalized cost)."""
    n = len(tasks)
    genotype = {t.id: rng.randrange(len(vms)) for t in tasks}
    makespan, cost, _ = decode(genotype, tasks, vms, avg_bw, transfer_size_bytes)

    fastest_mips = max(v.mips for v in vms)
    seq_m = sequential_makespan(tasks, fastest_mips)
    # cost of running the whole workflow sequentially on the cheapest VM,
    # used as the cost-normalization denominator for the same reason
    cheapest_cost_rate = min(v.cost_per_sec for v in vms)
    seq_c = seq_m * (fastest_mips / min(v.mips for v in vms)) * cheapest_cost_rate

    norm_makespan = makespan / seq_m
    norm_cost = cost / seq_c

    max_mips = max(v.mips for v in vms)
    max_cost_rate = max(v.cost_per_sec for v in vms)

    id_to_idx = {t.id: i for i, t in enumerate(task_order)}
    node_features = []
    for t in task_order:
        vm = vms[genotype[t.id]]
        node_features.append(struct_feats[t.id] + [vm.mips / max_mips, vm.cost_per_sec / max_cost_rate])

    edges = []
    for t in tasks:
        for c in t.children:
            edges.append((id_to_idx[t.id], id_to_idx[c.id]))

    return node_features, edges, norm_makespan, norm_cost


def load_workflow(path):
    tasks = parse_dax(path)
    task_order, depth = decode_order(tasks)
    struct_feats = build_graph_features(tasks, task_order, depth)
    vms = build_vms()
    avg_bw = average_bandwidth(vms)
    return tasks, task_order, depth, struct_feats, vms, avg_bw
