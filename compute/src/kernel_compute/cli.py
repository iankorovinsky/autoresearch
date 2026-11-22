import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

from kernel_compute.common import TERMINAL, digest
from kernel_compute.models import WaterlooConfig
from kernel_compute.providers import environment_lock
from kernel_compute.providers.base import ProviderError
from kernel_compute.providers.modal_backend import ModalConfig, ModalProvider
from kernel_compute.providers.waterloo_datasci import WaterlooProvider
from kernel_compute.service import ExecutionService
from kernel_compute.store import RunStore

PROBE_PROVIDERS = ("modal",)
PROVIDERS = ("waterloo-datasci", *PROBE_PROVIDERS)


def provider_factory(provider_id, config):
    if provider_id == "modal":
        return ModalProvider(ModalConfig.model_validate(config))
    if provider_id != "waterloo-datasci":
        raise ValueError(f"unknown provider: {provider_id}")
    return WaterlooProvider(WaterlooConfig.model_validate(config))


def configuration(args, provider_id=None):
    provider_id = provider_id or args.provider
    path = Path(args.config).expanduser()
    config = (
        tomllib.loads(path.read_text()).get("providers", {}).get(provider_id, {})
        if path.exists()
        else {}
    )
    if provider_id == "waterloo-datasci" and os.environ.get("KCOMPUTE_SSH_USER"):
        config["user"] = os.environ["KCOMPUTE_SSH_USER"]
    if getattr(args, "gpu", None) is not None:
        if provider_id not in PROBE_PROVIDERS:
            raise ValueError("--gpu requires --provider modal")
        config["gpu"] = args.gpu
    return provider_factory(provider_id, config)


def parser():
    root = argparse.ArgumentParser(description="Queue and profile GPU kernels")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON (default output)"
    )
    shared.add_argument(
        "--config", default=os.environ.get("KCOMPUTE_CONFIG", "~/.config/kcompute/config.toml")
    )
    shared.add_argument("--state-dir", type=Path)
    shared.add_argument("--provider", choices=PROVIDERS)
    subs = root.add_subparsers(dest="command", required=True)
    providers = subs.add_parser("providers", parents=[shared])
    providers.add_argument("action", choices=["list"])
    gpu_selection = argparse.ArgumentParser(add_help=False)
    gpu_selection.add_argument("--gpu", help="GPU type for Modal (see providers list)")
    subs.add_parser("doctor", parents=[shared, gpu_selection])
    subs.add_parser(
        "probe", parents=[shared, gpu_selection], help="test tools on a Modal GPU"
    )
    setup = subs.add_parser("setup", parents=[shared])
    setup.add_argument("--wait", action="store_true")
    setup.add_argument("--wait-timeout", type=float, default=1500)
    for name in ("submit", "run"):
        command = subs.add_parser(name, parents=[shared, gpu_selection])
        command.add_argument("manifests", type=Path, nargs="+" if name == "submit" else 1)
        command.add_argument("--attempt-of", help="link an explicit new attempt to an earlier run")
        if name == "run":
            command.add_argument("--wait-timeout", type=float)
            command.add_argument("--poll-interval", type=float, default=15)
    resume = subs.add_parser(
        "resume", parents=[shared], help="resume a recorded batch without duplicate submissions"
    )
    resume.add_argument("batch_id")
    for name in ("status", "result", "cancel", "logs"):
        command = subs.add_parser(name, parents=[shared])
        command.add_argument("run_id")
        if name == "cancel":
            command.add_argument("--wait", action="store_true")
            command.add_argument("--wait-timeout", type=float, default=120)
        if name == "logs":
            command.add_argument("--follow", action="store_true")
    subs.add_parser("list", parents=[shared])
    return root


def emit(value):
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if getattr(args, "poll_interval", 15) <= 0:
            raise ValueError("poll interval must be positive")
        if getattr(args, "wait_timeout", None) is not None and args.wait_timeout < 0:
            raise ValueError("wait timeout must be nonnegative")
        if args.command == "providers":
            provider_ids = [args.provider] if args.provider else PROVIDERS
            emit([configuration(args, provider_id).describe() for provider_id in provider_ids])
            return 0
        if args.provider is None:
            args.provider = "modal" if args.command == "probe" else "waterloo-datasci"
        if args.command in {"doctor", "setup", "probe", "submit", "run"}:
            provider = configuration(args)
        if args.command == "probe":
            if provider.id not in PROBE_PROVIDERS:
                raise ValueError("probe requires --provider modal")
            emit(provider.probe())
            return 0
        if args.command == "doctor":
            result = provider.doctor()
            emit(result)
            if provider.id in PROBE_PROVIDERS:
                return 0 if result["ready"] else 1
            return 0 if result["environment_ready"] and result["node"] else 1
        if args.command == "setup":
            result = provider.setup()
            if args.wait:
                deadline = time.monotonic() + args.wait_timeout
                print(f"Environment setup: {result}", file=sys.stderr, flush=True)
                while not result["ready"]:
                    if time.monotonic() >= deadline:
                        emit(result)
                        return 124
                    time.sleep(15)
                    result = provider.setup_status()
                    if not result["ready"] and any(
                        r["raw_state"].split()[0]
                        not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
                        for r in result["records"]
                    ):
                        emit(result)
                        return 1
            emit(result)
            return 0
        store = RunStore(args.state_dir)
        service = ExecutionService(store, provider_factory)
        if args.command in {"submit", "run"}:
            if args.attempt_of:
                store.get(args.attempt_of)
            batch = service.prepare(
                args.manifests, provider, digest(environment_lock()), args.attempt_of
            )
            print(
                f"Saved batch {batch['batch_id']}; runs: {', '.join(batch['run_ids'])}",
                file=sys.stderr,
                flush=True,
            )
            response = service.submit_batch(batch["batch_id"])
            if args.command == "submit":
                if len(response["runs"]) == 1:
                    emit({"batch_id": batch["batch_id"], **response["runs"][0]})
                else:
                    emit(response)
                return (
                    0
                    if all(
                        r["handle"] and r["state"] not in {"failed", "submitting"}
                        for r in response["runs"]
                    )
                    else 1
                )
            run_id = batch["run_ids"][0]
            service.wait(run_id, args.wait_timeout, args.poll_interval)
            result = service.result(run_id)
            emit(result)
            return 0 if result["execution"]["status"] == "succeeded" else 1
        if args.command == "resume":
            response = service.submit_batch(args.batch_id)
            emit(response)
            return (
                0
                if all(
                    r["handle"] and r["state"] not in {"failed", "submitting"}
                    for r in response["runs"]
                )
                else 1
            )
        if args.command == "status":
            emit(service.summary(service.status(args.run_id)))
        elif args.command == "result":
            emit(service.result(args.run_id))
        elif args.command == "cancel":
            result = service.cancel(args.run_id)
            if args.wait:
                result["final"] = service.summary(service.wait(args.run_id, args.wait_timeout))
            emit(result)
        elif args.command == "list":
            emit([service.summary(record) for record in store.list(args.provider)])
        elif args.command == "logs":
            cursor = None
            while True:
                value = service.logs(args.run_id, cursor)
                if args.json:
                    # JSON Lines for follow mode; each event includes a continuation cursor.
                    print(json.dumps(value), flush=True)
                else:
                    for name, chunk in value["chunks"].items():
                        if chunk:
                            print(f"[{name}]\n{chunk}", end="", flush=True)
                previous, cursor = cursor, value["cursor"]
                if not args.follow:
                    break
                terminal = service.status(args.run_id)["state"] in TERMINAL
                if terminal and previous == cursor:
                    break
                time.sleep(0.1 if terminal else 15)
        return 0
    except KeyboardInterrupt:
        print(
            "Detached. Submitted jobs continue; use the saved run ID or kcompute list.",
            file=sys.stderr,
        )
        return 130
    except TimeoutError as exc:
        emit({"error": str(exc), "kind": "wait_timeout"})
        return 124
    except (ValueError, OSError, ProviderError) as exc:
        emit({"error": str(exc), "kind": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
