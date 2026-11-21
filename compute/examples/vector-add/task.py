"""Compile a real CUDA kernel, launch on PyTorch's current stream, check against torch."""

import ctypes
import os
import subprocess
from pathlib import Path

import torch


class VectorAdd:
    mutates_inputs = False
    stateful = False
    timing_scope = "callable_current_stream"

    def __init__(self, case, device):
        self.n, self.device = int(case["n"]), device
        if not 0 < self.n < 2**31 - 255:
            raise ValueError("n outside supported int32 launch range")
        build_dir = Path("build").resolve()
        build_dir.mkdir(exist_ok=True)
        library = build_dir / "vector_add.so"
        major, minor = torch.cuda.get_device_capability(device)
        subprocess.run(
            [
                os.environ["CUDA_HOME"] + "/bin/nvcc",
                "-shared",
                "-Xcompiler=-fPIC",
                "-O3",
                f"-arch=sm_{major}{minor}",
                str(Path(__file__).with_name("kernel.cu")),
                "-o",
                str(library),
            ],
            check=True,
        )
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.launch_add
        self.launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def make_inputs(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return tuple(torch.randn(self.n, device=self.device, generator=generator) for _ in range(2))

    def run(self, a, b):
        output = torch.empty_like(a)
        code = self.launch(
            a.data_ptr(),
            b.data_ptr(),
            output.data_ptr(),
            self.n,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if code:
            raise RuntimeError(f"CUDA launch error {code}")
        return output

    def reference(self, a, b):
        return a + b


def build(case, device):
    return VectorAdd(case, device)
