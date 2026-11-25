"""Reference calibration with the unchanged validator (probe_habitat_flow_v1.references); no model outputs."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from suite_common import read, digest, frozen
from hc2_common import root, REF_ANGLES


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study', type=Path, required=True); a = ap.parse_args()
    out = root(a.study); freeze = read(out / 'execution_freeze.json')
    for path, sha in freeze['source_hashes'].items(): assert digest(path) == sha, path
    assert read(out / 'render_complete.json')['status'] == 'passed'
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from probe_habitat_flow_v1 import FlowStore, references
    flow = out / 'flow'; flow.mkdir(exist_ok=True); engine = FlowStore(flow)
    stub = out / 'reference_audit_input'; folder = stub / 'v2/test_references'; folder.mkdir(parents=True, exist_ok=True)
    frozen(stub / 'v2/eval_manifest.json', dict(angles=REF_ANGLES, source='history_confirmation_v2 references'))
    frozen(folder / 'manifest.json', read(out / 'references_manifest.json'))
    done = references(SimpleNamespace(study=stub), flow, engine)
    print('HC2_REFERENCE_GATES', read(flow / 'references/summary.json')['all_reference_gates_passed'], flush=True)
    return done


if __name__ == '__main__':
    main()
