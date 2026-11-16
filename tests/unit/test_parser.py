"""Unit tests for FastAPI Markdown parsing."""

from pathlib import Path

from fastapi_docs_rag.ingestion.parser import (
    guess_doc_type,
    parse_markdown,
    url_for,
)


def _make_doc(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path
    docs_root = repo_root / "docs" / "en" / "docs"
    tutorial = docs_root / "tutorial"
    tutorial.mkdir(parents=True)
    src = repo_root / "docs_src" / "first_steps"
    src.mkdir(parents=True)
    (src / "tutorial001.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n",
        encoding="utf-8",
    )
    doc_path = tutorial / "first-steps.md"
    doc_path.write_text(
        "\n".join(
            [
                "# First Steps { #first-steps }",
                "",
                "Intro.",
                "",
                "{* ../../docs_src/first_steps/tutorial001.py hl[1] *}",
                "",
                "## Deprecated Events { #deprecated-events }",
                "",
                "/// warning",
                "",
                "Use `lifespan` instead.",
                "",
                "///",
                "",
                "```python",
                "# this is a code comment, not a heading",
                "x = 1",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return repo_root, docs_root, doc_path


def test_parse_document_with_include_and_admonition(tmp_path: Path) -> None:
    repo_root, docs_root, doc_path = _make_doc(tmp_path)

    doc = parse_markdown(
        doc_path.read_text(encoding="utf-8"),
        doc_path=doc_path,
        docs_root=docs_root,
        repo_root=repo_root,
    )

    assert doc.title == "First Steps"
    assert doc.url == "https://fastapi.tiangolo.com/tutorial/first-steps/"
    assert doc.doc_type == "tutorial"
    assert doc.code_includes == ["docs_src/first_steps/tutorial001.py"]
    assert "from fastapi import FastAPI" in doc.markdown
    assert "```python" in doc.markdown
    assert "> **WARNING**" in doc.markdown
    assert "Use `lifespan` instead." in doc.markdown
    assert [section.title for section in doc.sections] == [
        "First Steps",
        "Deprecated Events",
    ]
    assert doc.sections[1].is_deprecated is True
    assert all("code comment" not in section.title for section in doc.sections)


def test_include_from_source_tree_with_line_range(tmp_path: Path) -> None:
    repo_root, docs_root, doc_path = _make_doc(tmp_path)
    source = repo_root / "fastapi" / "openapi" / "docs.py"
    source.parent.mkdir(parents=True)
    source.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    doc = parse_markdown(
        "# Source Snippet\n\n{* ../../fastapi/openapi/docs.py ln[2:3] *}\n",
        doc_path=doc_path,
        docs_root=docs_root,
        repo_root=repo_root,
    )

    assert "line2" in doc.markdown
    assert "line3" in doc.markdown
    assert "line1" not in doc.markdown
    assert doc.code_includes == ["fastapi/openapi/docs.py"]


def test_missing_include_is_reported_without_crashing(tmp_path: Path) -> None:
    repo_root, docs_root, doc_path = _make_doc(tmp_path)
    text = "# Missing\n\n{* ../../docs_src/nope.py *}\n"

    doc = parse_markdown(
        text,
        doc_path=doc_path.with_name("missing.md"),
        docs_root=docs_root,
        repo_root=repo_root,
    )

    assert "missing code include" in doc.markdown
    assert doc.code_includes == []


def test_url_mapping() -> None:
    assert url_for(Path("index.md")) == "https://fastapi.tiangolo.com/"
    assert url_for(Path("tutorial/index.md")) == "https://fastapi.tiangolo.com/tutorial/"
    assert (
        url_for(Path("advanced/events.md"))
        == "https://fastapi.tiangolo.com/advanced/events/"
    )


def test_doc_type_classification() -> None:
    assert guess_doc_type(Path("tutorial/first-steps.md")) == "tutorial"
    assert guess_doc_type(Path("advanced/events.md")) == "advanced"
    assert guess_doc_type(Path("release-notes.md")) == "release_notes"
    assert guess_doc_type(Path("about/index.md")) == "meta"
    assert guess_doc_type(Path("features.md")) == "concept"


def test_admonition_variants_are_rendered(tmp_path: Path) -> None:
    repo_root, docs_root, doc_path = _make_doc(tmp_path)
    text = "\n".join(
        [
            "# Admonitions",
            "",
            "/// tip | Read more",
            "",
            "Titled tip body.",
            "",
            "///",
            "",
            "1. Step one.",
            "",
            "    /// note",
            "",
            "    Indented note body.",
            "",
            "    ///",
            "",
            "//// tab | Python 3.10+",
            "",
            "/// warning",
            "",
            "Nested warning body.",
            "",
            "///",
            "",
            "////",
            "",
        ]
    )

    doc = parse_markdown(
        text,
        doc_path=doc_path.with_name("admonitions.md"),
        docs_root=docs_root,
        repo_root=repo_root,
    )

    assert "> **TIP**" in doc.markdown
    assert "> *Read more*" in doc.markdown
    assert "    > **NOTE**" in doc.markdown
    assert "    > Indented note body." in doc.markdown
    assert "> **TAB**" in doc.markdown
    assert "> > **WARNING**" in doc.markdown
    assert not [
        line for line in doc.markdown.splitlines() if line.lstrip().startswith("///")
    ]


def test_deprecation_signals(tmp_path: Path) -> None:
    repo_root, docs_root, doc_path = _make_doc(tmp_path)
    text = "\n".join(
        [
            "# Deprecations",
            "",
            "## Deprecating parameters",
            "",
            "Use `deprecated=True`.",
            "",
            "## Deprecate a path operation",
            "",
            "Pass `deprecated=True` to the decorator.",
            "",
            "## Alternative Events (deprecated)",
            "",
            "Use `lifespan`.",
            "",
            "### Old child",
            "",
            "Old child body.",
            "",
            "## Old events",
            "",
            "/// warning | Deprecated `on_event`",
            "",
            "Use `lifespan` instead.",
            "",
            "///",
            "",
            "## Parameter hint",
            "",
            "/// warning | Deprecated `root_path` parameter",
            "",
            "This parameter is now deprecated.",
            "",
            "///",
            "",
            "## Keyword hint",
            "",
            "/// note | Deprecated `example` keyword",
            "",
            "The `example` keyword is now deprecated.",
            "",
            "///",
            "",
            "## WSGI note",
            "",
            "/// note",
            "",
            "WSGIMiddleware is now deprecated.",
            "",
            "///",
            "",
            "## Prose mention",
            "",
            "This parameter is now deprecated.",
            "",
            "## Code comment",
            "",
            "```python",
            "# deprecated but this is only a code comment",
            "x = 1",
            "```",
            "",
            "## Inline code",
            "",
            "The OpenAPI field is called `deprecated`.",
            "",
        ]
    )

    doc = parse_markdown(
        text,
        doc_path=doc_path.with_name("deprecations.md"),
        docs_root=docs_root,
        repo_root=repo_root,
    )
    sections = {section.title: section for section in doc.sections}

    # Guides for marking something as deprecated are not deprecated APIs.
    assert sections["Deprecating parameters"].is_deprecated is False
    assert sections["Deprecating parameters"].mentions_deprecation is True
    assert sections["Deprecating parameters"].deprecation_hint is True
    assert sections["Deprecate a path operation"].is_deprecated is False
    assert sections["Deprecate a path operation"].deprecation_hint is True

    # Real section-level deprecations stay strong, including child sections.
    assert sections["Alternative Events (deprecated)"].is_deprecated is True
    assert sections["Old child"].is_deprecated is True
    assert sections["Old events"].is_deprecated is True
    assert sections["Old events"].deprecation_hint is False
    assert sections["WSGI note"].is_deprecated is True

    # Parameter/keyword notices are hints, not whole-section deprecations.
    assert sections["Parameter hint"].is_deprecated is False
    assert sections["Parameter hint"].deprecation_hint is True
    assert sections["Keyword hint"].is_deprecated is False
    assert sections["Keyword hint"].deprecation_hint is True

    assert sections["Prose mention"].is_deprecated is False
    assert sections["Prose mention"].mentions_deprecation is True
    assert sections["Code comment"].mentions_deprecation is False
    assert sections["Inline code"].mentions_deprecation is False
    assert doc.has_deprecated_section is True
    assert doc.has_deprecation_notes is True

    metadata = doc.metadata(
        docs_version="0.162.0",
        source_ref="deadbeef",
        fetched_at="2026-11-11",
        index_build_id="build-1",
    )
    assert metadata["has_deprecated_section"] is True
    assert metadata["has_deprecation_notes"] is True
