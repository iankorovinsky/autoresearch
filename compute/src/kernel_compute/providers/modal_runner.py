"""Sandbox entry point for a frozen kcompute bundle."""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main():
    mounted = Path("/kcompute-data")
    with tempfile.TemporaryDirectory(prefix="kcompute-") as root:
        with tarfile.open(mounted / "bundle.tar") as bundle:
            bundle.extractall(root, filter="data")
        env = {
            **os.environ,
            "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.environ["PATH"])),
            "PYTHONPATH": str(Path(root) / "worker"),
            "TMPDIR": root,
        }
        completed = subprocess.run(
            [sys.executable, "-m", "kernel_compute.worker.main"], cwd=root, env=env
        )
        # Publish the completion manifest last. A failed launcher must not advertise
        # a completed worker result.
        for name in ("request.json", "result.json", "samples.json", "stdout.log", "stderr.log"):
            source = Path(root) / name
            if source.exists():
                shutil.copyfile(source, mounted / name)
        completion = Path(root) / "completion.json"
        if completion.exists():
            shutil.copyfile(completion, mounted / "completion.tmp")
            (mounted / "completion.tmp").replace(mounted / "completion.json")
        subprocess.run(["sync", str(mounted)], check=True)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
