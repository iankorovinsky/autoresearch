"""Training Job entry point for frozen kernel bundles and the GPU capability probe."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent
    checkpoint_dir = Path(os.environ["BT_CHECKPOINT_DIR"])
    if sys.argv[1] == "probe":
        report = subprocess.check_output([sys.executable, str(source / "probe.py")], timeout=300)
        (checkpoint_dir / "probe.json").write_bytes(report)
        return 0

    with tempfile.TemporaryDirectory(prefix="kcompute-") as temp:
        root = Path(temp)
        with tarfile.open(source / "bundle.tar") as archive:
            archive.extractall(root, filter="data")
        inventory = {
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            "lock_hash": hashlib.sha256((source / "worker.lock").read_bytes()).hexdigest(),
        }
        (Path(sys.executable).parent.parent / "ready.json").write_text(json.dumps(inventory))
        completed = subprocess.run(
            [sys.executable, "-m", "kernel_compute.worker.main"],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "worker"), "TMPDIR": temp},
        )
        # The worker always publishes a result, including for a failed workload.
        # Finish the training job successfully so Baseten syncs that result.
        completion = root / "completion.json"
        if not completion.is_file():
            return completed.returncode or 1
        for name in json.loads(completion.read_text())["files"]:
            shutil.copyfile(root / name, checkpoint_dir / name)
        shutil.copyfile(completion, checkpoint_dir / completion.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
