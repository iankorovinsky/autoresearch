"""Run a candidate against a local KernelBench task.

Example:
    python3 scripts/check_candidate.py 1 1 ./candidate.py
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
RESERVED_SETTINGS = {
    "ref_origin", "dataset_src", "level", "problem_id", "kernel_src_path", "eval_mode"
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_command(level: int, problem_id: int, kernel_path: Path, settings: list[str], eval_mode: str = "local", gpu: str = "T4") -> list[str]:
    """Build the evaluator invocation without importing GPU dependencies."""
    return [
        "uv", "run", "python", "scripts/run_and_check.py",
        "ref_origin=kernelbench", "dataset_src=local",
        f"level={level}", f"problem_id={problem_id}",
        f"kernel_src_path={kernel_path}", f"eval_mode={eval_mode}",
        *([f"gpu={gpu}"] if eval_mode == "modal" else []),
        *settings,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", type=positive_int)
    parser.add_argument("problem_id", type=positive_int)
    parser.add_argument("kernel_path", type=Path, help="candidate file defining ModelNew")
    parser.add_argument("--eval-mode", choices=("local", "modal"), default="local")
    parser.add_argument("--gpu", default="T4", help="Modal GPU type (default: T4)")
    parser.add_argument(
        "--setting", action="append", default=[], metavar="KEY=VALUE",
        help="additional evaluator setting; repeat as needed (e.g. backend=triton)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    args = parser.parse_args()

    kernel_path = args.kernel_path.expanduser().resolve()
    if not kernel_path.is_file():
        parser.error(f"candidate file does not exist: {kernel_path}")
    task_dir = REPO_ROOT / "KernelBench" / f"level{args.level}"
    if not list(task_dir.glob(f"{args.problem_id}_*.py")):
        parser.error(f"no local task for level {args.level}, problem {args.problem_id}")
    for setting in args.setting:
        key, separator, value = setting.partition("=")
        if not separator or not key or not value:
            parser.error(f"expected KEY=VALUE, got {setting!r}")
        if key in RESERVED_SETTINGS:
            parser.error(f"{key} is controlled by the wrapper and cannot be overridden")

    command = build_command(args.level, args.problem_id, kernel_path, args.setting, args.eval_mode, args.gpu)
    env = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (source_path, env.get("PYTHONPATH")) if path
    )
    print(f"Working directory: {REPO_ROOT}", flush=True)
    print(f"PYTHONPATH={shlex.quote(env['PYTHONPATH'])}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    try:
        return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode
    except FileNotFoundError:
        parser.exit(127, "error: uv was not found on PATH; install uv before running evaluation\n")


if __name__ == "__main__":
    raise SystemExit(main())
