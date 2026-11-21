import argparse
import json
from dataclasses import asdict
from pathlib import Path

from repo_dev import docs
from repo_dev.doctor import diagnose
from repo_dev.tasks import COMPONENTS, catalog, execute, select_tasks


def parser():
    cli = argparse.ArgumentParser(description="Isolated repository development utilities")
    cli.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="machine-readable result; task logs go to stderr"
    )
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("list", parents=[common], help="list available tasks")
    for phase in ("setup", "check", "test", "run"):
        command = commands.add_parser(phase, parents=[common])
        if phase == "run":
            command.add_argument("task", choices=sorted(catalog()))
        else:
            command.add_argument(
                "components",
                nargs="*",
                metavar="COMPONENT",
                help="tooling compute frontend data; default: all",
            )
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--fail-fast", action="store_true")
    doctor = commands.add_parser(
        "doctor", parents=[common], help="read-only environment/configuration checks"
    )
    doctor.add_argument("components", nargs="*", metavar="COMPONENT")
    doctor.add_argument("--profile", choices=("local", "app"), default="local")
    doctor.add_argument(
        "--ssh", action="store_true", help="also check Waterloo access without submitting GPU jobs"
    )
    doctor.add_argument("--strict", action="store_true", help="fail on warnings as well as errors")
    document = commands.add_parser(
        "docs", parents=[common], help="offline documentation and artifact checks"
    )
    document.add_argument(
        "paths", nargs="*", help="repo-relative files/directories; indexes are always checked"
    )
    document.add_argument(
        "--include-sources", action="store_true", help="also inspect preserved paper Markdown"
    )
    document.add_argument(
        "--prose-paths", action="store_true", help="warn about unresolved code-formatted paths"
    )
    document.add_argument(
        "--strict", action="store_true", help="fail on warnings as well as errors"
    )
    return cli


def display(value, json_output):
    if json_output:
        print(json.dumps(value, indent=2, allow_nan=False))
    elif "tasks" in value:
        for task in value["tasks"]:
            if "status" in task:
                print(
                    f"{task['status'].upper():8} {task['task']}"
                    + (f": {task['message']}" if task.get("message") else "")
                )
                if task["status"] == "planned":
                    import shlex

                    print(f"         {task['cwd']}: {shlex.join(task['command'])}")
            else:
                print(f"{task['name']:20} {task['description']}")
    elif "checks" in value:
        for check in value["checks"]:
            print(f"{check['status'].upper():8} {check['id']}: {check['message']}")
    elif "findings" in value:
        for finding in value["findings"]:
            print(
                f"{finding['path']}:{finding['line']}: {finding['severity']} [{finding['code']}] {finding['message']}"
            )
        print(
            f"Checked {value['files_checked']} files: {value['errors']} errors, {value['warnings']} warnings"
        )
    else:
        print(value.get("error", str(value)))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        root = args.root.resolve()
        if not (root / "AGENTS.md").is_file() or not (root / "tooling/pyproject.toml").is_file():
            raise ValueError("not an autoresearch repository root; pass --root before the command")
        if args.command == "list":
            value = {"ok": True, "tasks": [asdict(task) for task in catalog().values()]}
        elif args.command in {"setup", "check", "test", "run"}:
            selected = (
                [catalog()[args.task]]
                if args.command == "run"
                else select_tasks(args.command, args.components)
            )
            value = execute(
                selected,
                root,
                dry_run=args.dry_run,
                json_output=args.json,
                fail_fast=args.fail_fast,
            )
        elif args.command == "doctor":
            if any(component not in COMPONENTS for component in args.components):
                raise ValueError("choose components from: " + ", ".join(COMPONENTS))
            value = diagnose(root, args.components, profile=args.profile, ssh=args.ssh)
            if args.strict and any(item["status"] == "warning" for item in value["checks"]):
                value["ok"] = False
        else:
            value = docs.check(
                root, args.paths, include_sources=args.include_sources, prose_paths=args.prose_paths
            )
            if args.strict and value["warnings"]:
                value["ok"] = False
        display(value, args.json)
        return 0 if value["ok"] else 1
    except KeyboardInterrupt:
        display({"ok": False, "error": "Interrupted; child task stopped"}, args.json)
        return 130
    except (OSError, ValueError) as exc:
        display({"ok": False, "error": str(exc)}, args.json)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
