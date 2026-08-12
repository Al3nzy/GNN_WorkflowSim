"""
Synthetic DAG generator, used purely to diversify training-data STRUCTURE.

The diagnosed problem: a GNN trained on only 4 real workflow families
(each with one characteristic DAG shape) can learn family-specific
structural shortcuts rather than transferable graph reasoning, which is
exactly why it generalises well to a bigger/smaller version of a familiar
shape (scale holdout) but fails on a genuinely new shape (family holdout).

This generates random DAGs with varied layering patterns: some wide and
shallow (Montage/CyberShake-like), some deep and narrow (Sipht-like),
some with irregular per-level width, spanning a wider region of
"DAG shape space" than the 5 real families alone occupy. Task lengths
and file transfer sizes are drawn from distributions matching the real
workflows' observed ranges, so the SIMULATION semantics (decoder, cost
model) stay grounded in the same reality; only the graph topology itself
is synthetic.
"""
import random
from dax_parser import DaxTask


def make_synthetic_dag(rng, n_tasks, shape='mixed'):
    """shape in {'wide_shallow', 'deep_narrow', 'irregular', 'mixed'}"""
    if shape == 'mixed':
        shape = rng.choice(['wide_shallow', 'deep_narrow', 'irregular'])

    if shape == 'wide_shallow':
        n_levels = max(2, int(n_tasks ** 0.3))
    elif shape == 'deep_narrow':
        n_levels = max(3, int(n_tasks * 0.6))
    else:  # irregular
        n_levels = max(2, rng.randint(int(n_tasks ** 0.25), int(n_tasks ** 0.75) + 1))

    # distribute n_tasks across n_levels, with some randomness in level width
    level_sizes = []
    remaining = n_tasks
    for lvl in range(n_levels - 1):
        if shape == 'irregular':
            w = max(1, int(rng.gauss(remaining / (n_levels - lvl), remaining / (n_levels - lvl) * 0.5)))
        else:
            w = max(1, remaining // (n_levels - lvl))
        w = min(w, remaining - (n_levels - lvl - 1))
        level_sizes.append(w)
        remaining -= w
    level_sizes.append(max(1, remaining))

    tasks = []
    levels = []
    tid = 0
    for lvl, size in enumerate(level_sizes):
        level_tasks = []
        for _ in range(size):
            length = max(100.0, rng.gauss(5000, 3000))
            t = DaxTask(str(tid), f'synthetic_{tid}', length)
            tasks.append(t)
            level_tasks.append(t)
            tid += 1
        levels.append(level_tasks)

    # wire edges: each task in level L+1 connects to 1-3 random tasks in level L
    for lvl in range(1, len(levels)):
        for child in levels[lvl]:
            n_parents = min(len(levels[lvl - 1]), rng.choice([1, 1, 1, 2, 2, 3]))
            parents = rng.sample(levels[lvl - 1], n_parents)
            for p in parents:
                p.children.append(child)
                child.parents.append(p)
                fname = f'f_{p.id}_{child.id}'
                size = max(1.0, rng.gauss(2e6, 3e6))
                p.output_files[fname] = size
                child.input_files[fname] = size

    return tasks


def build_synthetic_batch(rng, n_samples, size_range=(20, 300)):
    """Returns n_samples synthetic (tasks) DAGs of varied size and shape."""
    batch = []
    for _ in range(n_samples):
        n_tasks = rng.randint(*size_range)
        tasks = make_synthetic_dag(rng, n_tasks, shape='mixed')
        batch.append(tasks)
    return batch
