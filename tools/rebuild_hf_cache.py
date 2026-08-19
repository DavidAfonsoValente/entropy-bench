"""Recreate the HuggingFace cache symlinks after a GCS transfer.

`gcloud storage rsync` copies the cache's blobs/ but silently skips the snapshots/
symlinks that point into them, leaving a cache transformers cannot read. This rebuilds
snapshots/<rev>/<file> -> ../../blobs/<sha> plus refs/, from the manifest produced
alongside the upload. Avoids re-uploading the weights.

  python3 rebuild_hf_cache.py --manifest hub_manifest.json --hub $HF_HOME/hub
"""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--hub", required=True)
    p.add_argument("--only", default=None, help="restrict to one models-- dir")
    a = p.parse_args()

    man = json.load(open(a.manifest))
    hub = Path(a.hub)
    for model, info in man.items():
        if a.only and model != a.only:
            continue
        mroot = hub / model
        if not (mroot / "blobs").is_dir():
            print(f"[skip] {model}: no blobs/ present")
            continue
        snap = mroot / "snapshots" / info["rev"]
        snap.mkdir(parents=True, exist_ok=True)
        made = missing = 0
        for name, sha in info["files"].items():
            if sha == "__REGULAR__":
                continue
            blob = mroot / "blobs" / sha
            link = snap / name
            if not blob.exists():
                print(f"[warn] {model}: blob missing for {name} ({sha})")
                missing += 1
                continue
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(os.path.relpath(blob, snap), link)
            made += 1
        refs = mroot / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        for ref, val in info.get("refs", {}).items():
            (refs / ref).write_text(val)
        print(f"[ok] {model}: {made} links, {missing} missing, rev {info['rev']}")


if __name__ == "__main__":
    main()
