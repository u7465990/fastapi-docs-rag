"""Unit tests for parent/child chunking."""

from pathlib import Path

from langchain_rag.indexing.chunker import chunk_document, count_tokens
from langchain_rag.ingestion.parser import ParsedDocument, parse_markdown


def _doc(tmp_path: Path, text: str) -> ParsedDocument:
    docs_root = tmp_path / "docs" / "en" / "docs"
    doc_path = docs_root / "tutorial" / "sample.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text, encoding="utf-8")
    return parse_markdown(
        text,
        doc_path=doc_path,
        docs_root=docs_root,
        repo_root=tmp_path,
    )


def test_token_estimate_rounds_up() -> None:
    assert count_tokens("") == 1
    assert count_tokens("abcd") == 1
    assert count_tokens("abcde") == 2


def test_code_block_is_never_split(tmp_path: Path) -> None:
    functions = "\n\n".join(
        f"def fn_{i}():\n    value = {i}\n    return value" for i in range(8)
    )
    code = f"```python\n{functions}\n```"
    prose = " ".join(f"Sentence number {i} explains the handler." for i in range(30))
    doc = _doc(tmp_path, f"# Page\n\n## Handlers\n\n{prose}\n\n{code}\n")

    chunks = chunk_document(doc, parent_tokens=300, child_tokens=120, overlap_tokens=30)
    children = [chunk for chunk in chunks if chunk.level == "child"]

    assert len(children) >= 2
    assert sum("def fn_0" in chunk.text for chunk in children) == 1
    for child in children:
        assert child.text.count("```") % 2 == 0


def test_long_code_block_splits_on_definitions(tmp_path: Path) -> None:
    functions = "\n\n".join(
        f"def fn_{i}():\n    value = {i}\n    return value" for i in range(90)
    )
    doc = _doc(tmp_path, f"# Page\n\n## Code\n\n```python\n{functions}\n```\n")

    chunks = chunk_document(doc, parent_tokens=2000, child_tokens=400, overlap_tokens=0)
    code_chunks = [
        chunk for chunk in chunks if chunk.level == "child" and "def fn_" in chunk.text
    ]

    assert len(code_chunks) > 1
    assert sum(chunk.text.count("def fn_") for chunk in code_chunks) == 90
    for chunk in code_chunks:
        assert chunk.text.startswith("```python")
        assert chunk.text.rstrip().endswith("```")
        assert chunk.code_language == "python"
        assert chunk.content_type == "code"
        assert chunk.token_count <= 800


def test_heading_header_parent_link_and_url(tmp_path: Path) -> None:
    doc = _doc(tmp_path, "# First Steps\n\n## Create It { #create-it }\n\nHello world.\n")

    chunks = chunk_document(doc)
    parents = {chunk.chunk_id for chunk in chunks if chunk.level == "parent"}
    child = next(chunk for chunk in chunks if chunk.level == "child")

    assert child.parent_id in parents
    assert child.embed_text.startswith("[FastAPI > First Steps > Create It]\n")
    assert child.text == "Hello world."
    assert child.url == "https://fastapi.tiangolo.com/tutorial/sample/#create-it"
    assert child.anchor == "create-it"


def test_child_budget_and_overlap(tmp_path: Path) -> None:
    sentences = " ".join(
        f"Sentence {i} describes how FastAPI resolves dependencies." for i in range(30)
    )
    doc = _doc(tmp_path, f"# Page\n\n## Details\n\n{sentences}\n")

    chunks = chunk_document(doc, parent_tokens=800, child_tokens=80, overlap_tokens=30)
    children = [chunk for chunk in chunks if chunk.level == "child"]

    assert len(children) > 2
    assert all(chunk.token_count <= 80 for chunk in children)
    for previous, current in zip(children, children[1:], strict=False):
        assert set(previous.text.splitlines()) & set(current.text.splitlines())


def test_code_inside_admonition_is_atomic(tmp_path: Path) -> None:
    code = "```python\n" + "\n".join(f"x_{i} = {i}" for i in range(40)) + "\n```"
    doc = _doc(tmp_path, f"# Page\n\n## Tips\n\n/// tip\n\n{code}\n\n///\n")

    chunks = chunk_document(doc, parent_tokens=300, child_tokens=100, overlap_tokens=20)
    children = [chunk for chunk in chunks if chunk.level == "child"]

    assert sum("x_0" in chunk.text for chunk in children) == 1
    for child in children:
        assert child.text.count("```") % 2 == 0


def test_deprecated_section_propagates(tmp_path: Path) -> None:
    text = (
        "# Page\n\n## Old Events\n\n/// warning\n\n"
        "`on_event` is deprecated.\n\n///\n\nUse `lifespan` instead.\n"
    )
    doc = _doc(tmp_path, text)

    chunks = chunk_document(doc)

    assert chunks
    assert all(chunk.is_deprecated for chunk in chunks)


def test_content_type_classification(tmp_path: Path) -> None:
    text = (
        "# Page\n\n## Prose\n\nJust words here.\n\n"
        "## Code\n\n```python\nx = 1\n```\n\n"
        "## Both\n\nSome words.\n\n```python\ny = 2\n```\n"
    )
    doc = _doc(tmp_path, text)

    children = {
        chunk.section_path[-1]: chunk.content_type
        for chunk in chunk_document(doc)
        if chunk.level == "child"
    }

    assert children == {"Prose": "prose", "Code": "code", "Both": "mixed"}


def test_chunk_ids_are_stable_and_unique(tmp_path: Path) -> None:
    doc = _doc(tmp_path, "# Page\n\n## One\n\nAlpha.\n\n## Two\n\nBeta.\n")

    first = chunk_document(doc)
    second = chunk_document(doc)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(chunk.level == "parent" or chunk.parent_id for chunk in first)
