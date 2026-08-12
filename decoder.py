"""Insertion-based decoder: identical logic to HEFTPlanningAlgorithm's own
gap-filling scheduler, validated earlier in this project against real
WorkflowSim source. Reused here unchanged so training data reflects the
exact same simulator semantics LIWSA/LIWSA-ML are evaluated under."""


class VM:
    def __init__(self, vm_id, mips, bw_mbit_s, cost_per_sec):
        self.id = vm_id
        self.mips = mips
        self.bw = bw_mbit_s
        self.cost_per_sec = cost_per_sec


def average_bandwidth(vms):
    return sum(v.bw for v in vms) / len(vms)


def transfer_time(size_bytes, avg_bw_mbit_s):
    return (size_bytes / 1e6) * 8.0 / avg_bw_mbit_s


def decode_order(tasks):
    depth = {}

    def d(t):
        if t.id in depth:
            return depth[t.id]
        depth[t.id] = 0 if not t.parents else 1 + max(d(p) for p in t.parents)
        return depth[t.id]

    for t in tasks:
        d(t)
    return sorted(tasks, key=lambda t: (depth[t.id], -t.length, t.id)), depth


def find_finish_time(sched_events, ready_time, duration, commit):
    if not sched_events:
        if commit:
            sched_events.append((ready_time, ready_time + duration))
        return ready_time + duration

    if len(sched_events) == 1:
        s0, f0 = sched_events[0]
        if ready_time >= f0:
            pos, start = 1, ready_time
        elif ready_time + duration <= s0:
            pos, start = 0, ready_time
        else:
            pos, start = 1, f0
        if commit:
            sched_events.insert(pos, (start, start + duration))
        return start + duration

    start = max(ready_time, sched_events[-1][1])
    finish = start + duration
    pos = len(sched_events)
    i = len(sched_events) - 1
    j = len(sched_events) - 2
    while j >= 0:
        cur_s, cur_f = sched_events[i]
        prev_s, prev_f = sched_events[j]
        if ready_time > prev_f:
            if ready_time + duration <= cur_s:
                start = ready_time
                finish = ready_time + duration
            break
        if prev_f + duration <= cur_s:
            start = prev_f
            finish = prev_f + duration
            pos = i
        i -= 1
        j -= 1

    if ready_time + duration <= sched_events[0][0]:
        pos, start = 0, ready_time
        if commit:
            sched_events.insert(pos, (start, start + duration))
        return start + duration

    if commit:
        sched_events.insert(pos, (start, finish))
    return finish


def decode(assignment, tasks, vms, avg_bw_mbit_s, transfer_size_fn):
    order, _ = decode_order(tasks)
    finish = {}
    sched_events = {vm.id: [] for vm in vms}
    cost = 0.0

    for t in order:
        vm = vms[assignment[t.id]]
        ready = 0.0
        for p in t.parents:
            p_finish = finish[p.id]
            if assignment[p.id] != assignment[t.id]:
                ready = max(ready, p_finish + transfer_time(transfer_size_fn(p, t), avg_bw_mbit_s))
            else:
                ready = max(ready, p_finish)
        duration = t.length / vm.mips
        fin = find_finish_time(sched_events[vm.id], ready, duration, commit=True)
        finish[t.id] = fin
        cost += duration * vm.cost_per_sec

    makespan = max(finish.values())
    return makespan, cost, finish
