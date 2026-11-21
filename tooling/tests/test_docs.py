import json

import pytest

from repo_dev.docs import check, strict_json


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def codes(result):
    return {item["code"] for item in result["findings"]}


def test_markdown_links_references_anchors_and_code(tmp_path):
    write(
        tmp_path,
        "README.md",
        """# Home
[reference][target]
[space](<guide with spaces.md#duplicate-1>)
[root](/guide%20with%20spaces.md#custom)
![image](image.png)
<a href="guide%20with%20spaces.md#duplicate">HTML</a>
`[ignored](does-not-exist.md)`
```md
[also ignored](does-not-exist.md)
```
[external](https://invalid.example/never-requested)
[target]: guide%20with%20spaces.md#duplicate
""",
    )
    write(tmp_path, "guide with spaces.md", '# Duplicate\n# Duplicate\n<a id="custom"></a>')
    write(tmp_path, "image.png", "placeholder")
    assert check(tmp_path)["ok"]


def test_actual_missing_targets_and_fragments(tmp_path):
    write(
        tmp_path,
        "README.md",
        "[bad](missing.md)\n[anchor](present.md#no-such-heading)\n[escape](../outside.md)",
    )
    write(tmp_path, "present.md", "# Exists")
    result = check(tmp_path)
    assert codes(result) == {"missing-link", "missing-anchor", "link-outside-repo"}
    assert not result["ok"]


def test_artifact_index_covers_nested_markdown(tmp_path):
    write(tmp_path, "wiki/artifacts/README.md", "# Artifacts\n[one](nested/one.md)")
    write(tmp_path, "wiki/artifacts/nested/one.md", "# One")
    write(tmp_path, "wiki/artifacts/two.md", "# Two")
    result = check(tmp_path)
    assert result["errors"] == 1
    assert result["findings"][0]["path"].endswith("two.md")
    assert "unindexed-artifact" in codes(result)


def test_missing_index(tmp_path):
    write(tmp_path, "wiki/artifacts/new.md", "# New")
    assert "missing-index" in codes(check(tmp_path))


def test_jsonc_comments_strings_and_trailing_commas():
    text = """{
      // a comment
      "url": "https://host/path/*literal*/", /* another comment */
      "quoted": "a\\\"b",
      "list": [1,2,],
    }"""
    result = strict_json(text, jsonc=True)
    assert result["url"] == "https://host/path/*literal*/"
    assert result["quoted"] == 'a"b'
    assert result["list"] == [1, 2]
    with pytest.raises(ValueError):
        strict_json(text)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', "[Infinity]", '{"a":}', "{"])
def test_malformed_json_is_reported(tmp_path, text):
    write(tmp_path, "data/example.json", text)
    result = check(tmp_path)
    assert result["errors"] == 1 and "invalid-json" in codes(result)


def test_source_snapshots_and_generated_dirs_excluded(tmp_path):
    write(tmp_path, "wiki/papers/pdfs/source.md", "[lost-in-conversion](absent.md)")
    write(tmp_path, "frontend/node_modules/dep/README.md", "[bad](missing.md)")
    write(tmp_path, "frontend/.next/broken.json", "not json")
    assert check(tmp_path)["ok"]
    assert "missing-link" in codes(check(tmp_path, include_sources=True))


def test_catalog_paths_and_count(tmp_path):
    write(
        tmp_path,
        "wiki/papers/analysis/index.json",
        json.dumps({"entry_count": 2, "entries": [{"analysis_file": "absent.md"}]}),
    )
    assert codes(check(tmp_path)) == {"catalog-count", "catalog-path"}


def test_prose_paths_are_opt_in_warnings(tmp_path):
    write(tmp_path, "README.md", "Future `scripts/planned.py`.")
    assert check(tmp_path)["ok"]
    result = check(tmp_path, prose_paths=True)
    assert result["ok"] and result["warnings"] == 1
    assert "prose-path" in codes(result)


@pytest.mark.parametrize("text", ["```python\nprint(1)", "~~~\nexample"])
def test_unclosed_fence(tmp_path, text):
    write(tmp_path, "README.md", text)
    assert "unclosed-fence" in codes(check(tmp_path))


def test_targeted_checks_reject_escape_and_missing_paths(tmp_path):
    for path in ("../outside.md", "missing.md"):
        with pytest.raises(ValueError):
            check(tmp_path, [path])


def test_link_cannot_follow_symlink_outside_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    external = write(tmp_path, "external.md", "# External")
    (root / "link.md").symlink_to(external)
    write(root, "README.md", "[external](link.md)")
    assert "link-outside-repo" in codes(check(root))
