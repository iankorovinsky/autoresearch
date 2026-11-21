"""Self-contained ModelNew example; does not import KernelBench or Modal."""

import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return torch.clamp_min(x, 0)


class Task:
    mutates_inputs = False
    stateful = False
    timing_scope = "callable_current_stream"

    def __init__(self, case, device):
        self.n, self.device = int(case["n"]), device
        self.model = Model().to(device).eval()
        self.candidate = ModelNew().to(device).eval()
        self.candidate.load_state_dict(self.model.state_dict())

    def make_inputs(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return (torch.randn(self.n, device=self.device, generator=generator),)

    def run(self, x):
        return self.candidate(x)

    def reference(self, x):
        return self.model(x)


def build(case, device):
    return Task(case, device)
