"""
DAX parser matching org.workflowsim.WorkflowParser's actual parsing rules:
  - length (MI) = max(1000 * runtime_seconds, 100)
  - file size read as-is in bytes; 0 bumped to 1
  - dependencies from explicit <child ref><parent ref/></child> elements
"""
import xml.etree.ElementTree as ET


class DaxTask:
    def __init__(self, job_id, name, length):
        self.id = job_id
        self.name = name
        self.length = length
        self.parents = []
        self.children = []
        self.input_files = {}
        self.output_files = {}


def _local(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag


def parse_dax(path):
    tree = ET.parse(path)
    root = tree.getroot()
    by_ref = {}
    order = []

    for job in root:
        if _local(job.tag) != 'job':
            continue
        job_id = job.get('id')
        name = job.get('name')
        runtime_attr = job.get('runtime')
        length = max(1000.0 * float(runtime_attr), 100.0) if runtime_attr else 0.0
        task = DaxTask(job_id, name, length)

        for uses in job:
            if _local(uses.tag) != 'uses':
                continue
            fname = uses.get('name') or uses.get('file')
            link = uses.get('link')
            size_attr = uses.get('size')
            size = float(size_attr) if size_attr is not None else 0.0
            if size == 0:
                size = 1.0
            if link == 'input':
                task.input_files[fname] = size
            elif link == 'output':
                task.output_files[fname] = size

        by_ref[job_id] = task
        order.append(job_id)

    for child_el in root:
        if _local(child_el.tag) != 'child':
            continue
        child_task = by_ref[child_el.get('ref')]
        for parent_el in child_el:
            if _local(parent_el.tag) != 'parent':
                continue
            parent_task = by_ref[parent_el.get('ref')]
            child_task.parents.append(parent_task)
            parent_task.children.append(child_task)

    return [by_ref[r] for r in order]


def transfer_size_bytes(parent, child):
    total = 0.0
    for fname, size in parent.output_files.items():
        if fname in child.input_files:
            total += size
    return total
