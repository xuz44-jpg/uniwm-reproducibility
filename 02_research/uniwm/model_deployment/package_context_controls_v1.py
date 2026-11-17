"""Package only this complete inference study; exclude weights and flow caches."""
import argparse
from pathlib import Path
import tarfile
from suite_common import read,digest,atomic
from context_controls_common import root


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args();out=root(a.study)
    assert read(out/'complete.json')['status']=='passed'
    files=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(out)
        if any(x in rel.parts for x in ['flow','cache','__pycache__']):continue
        if p.suffix not in ['.json','.jsonl','.png','.npy','.npz','.md','.py']:continue
        if p.name in ['retrieval_manifest.json','package_complete.json']:continue
        files.append(p)
    atomic(out/'retrieval_manifest.json',dict(files={str(p.relative_to(out)):digest(p) for p in files},
        exclusions=['large model weights','flow caches','speech data'],conditions=984,new_predictions=816))
    files.append(out/'retrieval_manifest.json');archive=Path('output/input-context-controls-v1-complete.tar.gz')
    tmp=archive.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
    tmp.replace(archive)
    atomic(out/'package_complete.json',dict(status='passed',path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,files=len(files)))


if __name__=='__main__':main()
