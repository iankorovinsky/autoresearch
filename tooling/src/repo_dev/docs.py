"""Offline documentation checks. Preserved paper text is excluded by default."""

import json
import os
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

ROOTS = ("README.md", "AGENTS.md", "compute", "frontend", "data", "tooling", "wiki")
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".next",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    "coverage",
}
PARSER = MarkdownIt("commonmark").enable("table")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    message: str
    severity: str = "error"


class HTMLReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.anchors.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.anchors.add(attrs["name"])
        for attribute in ("href", "src"):
            if attrs.get(attribute):
                self.links.append(attrs[attribute])


def markdown_info(text):
    tokens = PARSER.parse(text)
    anchors, links, code_paths, fences = set(), [], [], []
    for index, token in enumerate(tokens):
        line = (token.map or [0])[0] + 1
        if token.type == "heading_open":
            heading = tokens[index + 1]
            title = "".join(
                child.content
                for child in heading.children or []
                if child.type in {"text", "code_inline", "image"}
            )
            slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            candidate, suffix = slug, 0
            while candidate in anchors:
                suffix += 1
                candidate = f"{slug}-{suffix}"
            anchors.add(candidate)
        if token.type == "fence":
            fences.append(token)
        nodes = token.children or [token]
        for child in nodes:
            if child.type in {"link_open", "image"}:
                links.append((child.attrGet("href") or child.attrGet("src"), line))
            elif child.type in {"html_inline", "html_block"}:
                html = HTMLReferences()
                html.feed(child.content)
                links.extend((href, line) for href in html.links)
                anchors.update(html.anchors)
            elif child.type == "code_inline":
                code_paths.append((child.content, line))
    return anchors, links, code_paths, fences


def strip_jsonc(text):
    """Strip comments and trailing commas outside strings, preserving line positions."""
    chars = list(text)
    index, quoted = 0, False
    while index < len(chars):
        char = chars[index]
        if quoted:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif text[index : index + 2] == "//":
            end = text.find("\n", index)
            end = len(chars) if end < 0 else end
            chars[index:end] = " " * (end - index)
            index = end
            continue
        elif text[index : index + 2] == "/*":
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated JSONC comment")
            end += 2
            for pos in range(index, end):
                if chars[pos] not in "\r\n":
                    chars[pos] = " "
            index = end
            continue
        index += 1
    index, quoted = 0, False
    while index < len(chars):
        char = chars[index]
        if quoted:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == ",":
            end = index + 1
            while end < len(chars) and chars[end].isspace():
                end += 1
            if end < len(chars) and chars[end] in "}]":
                chars[index] = " "
        index += 1
    return "".join(chars)


def strict_json(text, *, jsonc=False):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("non-finite JSON number")

    return json.loads(
        strip_jsonc(text) if jsonc else text, object_pairs_hook=pairs, parse_constant=constant
    )


def discover(root, selected=(), include_sources=False):
    paths = []
    for item in selected or ROOTS:
        start = (root / item).resolve()
        if not start.is_relative_to(root):
            raise ValueError("documentation paths must stay inside the repository")
        if not start.exists():
            if selected:
                raise ValueError(f"documentation path does not exist: {item}")
            continue
        if start.is_file():
            paths.append(start)
        else:
            for directory, dirs, files in os.walk(start, followlinks=False):
                dirs[:] = [
                    name
                    for name in dirs
                    if name not in SKIP_DIRS
                    and not name.startswith(".")
                    and not (Path(directory) / name).is_symlink()
                    and not (Path(directory) / name / ".git").exists()
                ]
                paths.extend(Path(directory) / name for name in files)
    result = []
    for path in sorted(set(paths)):
        if (
            path.is_symlink()
            or path.name.startswith(".")
            or path.suffix not in {".md", ".json", ".jsonc"}
        ):
            continue
        rel = path.relative_to(root)
        if (
            not include_sources
            and str(rel).startswith("wiki/papers/pdfs/")
            and path.suffix == ".md"
            and path.name != "README.md"
        ):
            continue
        result.append(path)
    return result


def local_target(root, source, href):
    url = urlsplit(href)
    if url.scheme or url.netloc:
        return None
    decoded = unquote(url.path)
    target = (
        root / decoded.lstrip("/")
        if decoded.startswith("/")
        else source.parent / decoded
        if decoded
        else source
    )
    return target.resolve(), unquote(url.fragment)


def check(root: Path, selected=(), *, include_sources=False, prose_paths=False):
    root = root.resolve()
    files = discover(root, selected, include_sources)
    findings = []
    parsed, documents = {}, {}

    def add(path, line, code, message, severity="error"):
        findings.append(Finding(str(path.relative_to(root)), line, code, message, severity))

    def info(path):
        if path not in documents:
            documents[path] = markdown_info(path.read_text())
        return documents[path]

    for path in files:
        try:
            text = path.read_text()
            if path.suffix in {".json", ".jsonc"}:
                parsed[path] = strict_json(
                    text, jsonc=path.suffix == ".jsonc" or path.name.startswith("tsconfig")
                )
                continue
            _, links, literals, fences = info(path)
            for href, line in links:
                target = local_target(root, path, href)
                if target is None:
                    continue
                destination, fragment = target
                if not destination.is_relative_to(root):
                    add(path, line, "link-outside-repo", "Local link escapes the repository")
                elif not destination.exists():
                    add(path, line, "missing-link", f"Missing local target: {href}")
                elif fragment and destination.suffix == ".md":
                    if fragment not in info(destination)[0]:
                        add(
                            path, line, "missing-anchor", f"Missing Markdown heading/anchor: {href}"
                        )
            lines = text.splitlines()
            for token in fences:
                last = lines[token.map[1] - 1] if token.map[1] else ""
                closing = (
                    r"^\s*(?:>\s*)*"
                    + re.escape(token.markup[0])
                    + "{"
                    + str(len(token.markup))
                    + r",}\s*$"
                )
                if token.map[1] - token.map[0] < 2 or not re.match(closing, last):
                    add(
                        path,
                        token.map[0] + 1,
                        "unclosed-fence",
                        "Fenced code block has no closing fence",
                    )
            if prose_paths:
                for literal, line in literals:
                    if re.fullmatch(
                        r"(?:[\w.-]+/)+[\w.-]+\.(?:py|md|json|toml|ts|tsx|sh)", literal
                    ):
                        candidates = (path.parent / literal, root / literal)
                        if not any(candidate.exists() for candidate in candidates):
                            add(
                                path,
                                line,
                                "prose-path",
                                f"Unresolved code-formatted path (may be an example/proposal): {literal}",
                                "warning",
                            )
        except (OSError, UnicodeError, ValueError) as exc:
            add(
                path,
                getattr(exc, "lineno", 1),
                "invalid-json" if path.suffix != ".md" else "unreadable-document",
                str(exc),
            )

    # Index checks always run, even for a targeted file check.
    artifact_root = root / "wiki/artifacts"
    index = artifact_root / "README.md"
    if artifact_root.exists():
        if not index.exists():
            add(index, 1, "missing-index", "Artifact README index is missing")
        else:
            indexed = set()
            for href, _ in info(index)[1]:
                target = local_target(root, index, href)
                if target:
                    indexed.add(target[0])
            for path in sorted(artifact_root.rglob("*.md")):
                if (
                    path.name != "README.md"
                    and not path.is_symlink()
                    and path.resolve() not in indexed
                ):
                    add(
                        path,
                        1,
                        "unindexed-artifact",
                        "Add this Markdown artifact to wiki/artifacts/README.md",
                    )

    for name, fields in (
        (
            "wiki/papers/analysis/index.json",
            ("analysis_file", "source_markdown", "source_pdf", "source_original_html"),
        ),
        ("wiki/papers/pdfs/manifest.json", ("markdown", "pdf", "original_html")),
    ):
        path = root / name
        if not path.exists():
            continue
        try:
            catalog = parsed.get(path) or strict_json(path.read_text())
            entries = catalog["entries"]
            if not isinstance(entries, list) or any(
                not isinstance(entry, dict) for entry in entries
            ):
                raise ValueError("entries must be an array of objects")
            if "entry_count" in catalog and catalog["entry_count"] != len(entries):
                add(path, 1, "catalog-count", "entry_count does not match entries")
            for entry in entries:
                for field in fields:
                    value = entry.get(field)
                    if value:
                        if not isinstance(value, str):
                            raise ValueError(f"{field} must be a string or null")
                        destination = (path.parent / value).resolve()
                        if not destination.is_relative_to(root) or not destination.is_file():
                            add(path, 1, "catalog-path", f"Missing/escaping {field}: {value}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not any(f.path == name and f.code == "invalid-json" for f in findings):
                add(path, 1, "invalid-catalog", str(exc))

    errors = sum(item.severity == "error" for item in findings)
    return {
        "ok": errors == 0,
        "files_checked": len(files),
        "errors": errors,
        "warnings": len(findings) - errors,
        "findings": [asdict(item) for item in findings],
        "scope": {
            "paper_source_markdown": include_sources,
            "external_urls": "not requested",
            "prose_paths": prose_paths,
        },
    }
