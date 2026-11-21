import math
import statistics


def summarize(samples):
    if len(samples) < 2 or any(not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError("expected at least two finite, nonnegative timing samples")
    return {
        "units": "us",
        "sample_count": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "stddev": statistics.pstdev(samples),
        "p95": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
    }


def check_output(torch, actual, expected, policy):
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    if len(actual) != len(expected) or not actual:
        raise AssertionError("output tuple length differs or is empty")
    for left, right in zip(actual, expected, strict=True):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError("only tensor or fixed tuple-of-tensor outputs are supported")
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            raise AssertionError("output shape, dtype, or device differs")
        if left.is_floating_point() or left.is_complex():
            torch.testing.assert_close(
                left, right, rtol=policy["rtol"], atol=policy["atol"], equal_nan=policy["equal_nan"]
            )
        elif not torch.equal(left, right):
            raise AssertionError("integer/Boolean outputs differ")


def measure(torch, task, inputs, settings):
    """Time callable GPU work on the current stream; no cache flush or CPU timer."""
    import time

    start_wall = time.perf_counter()
    for _ in range(settings["warmup_iterations"]):
        output = task.run(*inputs)
        del output
    torch.cuda.synchronize()
    warmup_ms = (time.perf_counter() - start_wall) * 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    overhead_us = start.elapsed_time(end) * 1000
    samples = []
    for _ in range(settings["sample_count"]):
        start.record()
        output = task.run(*inputs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
        del output
    summary = summarize(samples)
    summary.update(settings)
    summary["event_pair_overhead_us"] = overhead_us
    return summary, samples, warmup_ms


def memory(torch, task, inputs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    output = task.run(*inputs)
    torch.cuda.synchronize()
    result = {
        "scope": "pytorch_allocator_including_resident_inputs_and_model",
        "units": "bytes",
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }
    del output
    return result
