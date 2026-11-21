"""Executed on a Slurm compute node to create a hash-addressed worker environment."""

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
import venv
from pathlib import Path

PIP_URL = "https://files.pythonhosted.org/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/pip-25.3-py3-none-any.whl"
PIP_SHA256 = "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd"


def main():
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "ready.json").exists():
            print("Environment already ready", flush=True)
            return
        lock_hash = hashlib.sha256((root / "worker.lock").read_bytes()).hexdigest()
        venv.EnvBuilder(with_pip=False).create(root)
        wheel = root / "pip-25.3-py3-none-any.whl"
        data = urllib.request.urlopen(PIP_URL, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != PIP_SHA256:
            raise ValueError("pip bootstrap checksum mismatch")
        wheel.write_bytes(data)
        python = str(root / "bin/python")
        launcher = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('pip',run_name='__main__')"
        subprocess.run(
            [
                python,
                "-c",
                launcher,
                str(wheel),
                "install",
                "--require-hashes",
                "--only-binary=:all:",
                "--no-cache-dir",
                "--disable-pip-version-check",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu130",
                "-r",
                str(root / "worker.lock"),
            ],
            check=True,
        )
        smoke = """
import importlib.metadata, json, subprocess, torch
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
a = torch.arange(1024, device='cuda', dtype=torch.float32)
assert torch.equal(a + a, 2 * a)
torch.cuda.synchronize()
print(json.dumps({'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
 'torch_cuda': torch.version.cuda, 'python': __import__('platform').python_version(),
 'compute_capability': torch.cuda.get_device_capability(),
 'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))}))
"""
        ready = json.loads(subprocess.check_output([python, "-c", smoke], text=True))
        # Verify actual local-toolkit code generation on the same allocated GPU.
        source = root / "smoke.cu"
        source.write_text("""#include <cuda_runtime.h>
#include <stdio.h>
__global__ void fill(int *p) { *p = 7; }
int main() { int *p, h=0; if(cudaMalloc(&p,sizeof(int))) return 1;
fill<<<1,1>>>(p); if(cudaDeviceSynchronize()) return 2;
if(cudaMemcpy(&h,p,sizeof(int),cudaMemcpyDeviceToHost)) return 3;
cudaFree(p); return h == 7 ? 0 : 4; }
""")
        arch = "sm_" + "".join(map(str, ready["compute_capability"]))
        subprocess.run(
            [
                os.environ["CUDA_HOME"] + "/bin/nvcc",
                "-arch=" + arch,
                str(source),
                "-o",
                str(root / "smoke"),
            ],
            check=True,
        )
        subprocess.run([str(root / "smoke")], check=True, timeout=30)
        ready["lock_hash"] = lock_hash
        ready["cuda_home"] = os.environ["CUDA_HOME"]
        (root / "ready.tmp").write_text(json.dumps(ready, indent=2))
        (root / "ready.tmp").replace(root / "ready.json")
        print(json.dumps(ready, indent=2), flush=True)


if __name__ == "__main__":
    main()
