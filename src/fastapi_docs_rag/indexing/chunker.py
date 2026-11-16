"""Split parsed FastAPI documents into parent and child chunks.

A parent keeps a coherent run of sections for the LLM; children (~400 tokens)
are what gets embedded. Code fences are atomic: a fence is never cut
mid-block, and only a block larger than 800 tokens is split on blank lines or
definition boundaries, with the language marker kept on every piece.

Small FastAPI sections are common, so parents are packed across consecutive
sections up to ``parent_tokens`` instead of forcing one tiny parent per
section. Children never span section boundaries, which keeps citations precise.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from fastapi_docs_rag.ingestion.parser import (
    CODE_FENCE_RE,
    ParsedDocument,
    Section,
    _advance_fence,
)

#: English prose averages ~4 characters per token. This is a rough estimate,
#: not a tokenizer; code may be slightly under-counted.
TOKEN_CHARS = 4

#: A code block up to this size stays atomic even if it exceeds the child
#: budget: cutting code is worse than a slightly oversized chunk.
CODE_BLOCK_MAX_TOKENS = 800

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DEFINITION_RE = re.compile(r"^(?:async\s+)?(?:def|class)\s")
QUOTE_PREFIX_RE = re.compile(r"^\s*(?:>\s?)+")
VERSION_HEADING_RE = re.compile(
    r"^\d+\.\d+(?:\.\d+)?(?:[A-Za-z0-9.\-+]*)?(?:\s|$)"
)
LATEST_CHANGES_TITLE = "Latest Changes"

BlockKind = Literal["prose", "code"]
ContentType = Literal["prose", "code", "mixed"]
ChunkLevel = Literal["parent", "child"]


@dataclass(frozen=True)
class _Block:
    """An atomic unit of text: a paragraph or a whole fenced code block."""

    text: str
    kind: BlockKind = "prose"
    language: str | None = None
    #: Text inserted before this block when joining a run of blocks.
    separator: str = "\n\n"


@dataclass(frozen=True)
class _SectionBlocks:
    """A section plus the atomic blocks that belong to it."""

    section: Section
    blocks: list[_Block]
    tokens: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    parent_id: str
    level: ChunkLevel
    text: str
    embed_text: str
    token_count: int
    section_path: list[str]
    anchor: str | None
    url: str
    content_type: ContentType
    code_language: str | None
    is_deprecated: bool
    deprecation_hint: bool
    section_hash: str
    chunk_index: int


def count_tokens(text: str) -> int:
    """Approximate the number of LLM tokens in *text*."""
    return max(1, (len(text) + TOKEN_CHARS - 1) // TOKEN_CHARS)


LANGUAGE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+#._-]*$")


def _language_of(fence_line: str) -> str | None:
    match = CODE_FENCE_RE.match(fence_line)
    if not match:
        return None
    info = match.group(2).strip()
    if not info:
        return None
    token = info.split()[0]
    # Skip MkDocs attributes such as ``{ .dockerfile }`` or ``hl_lines="4"``.
    if not LANGUAGE_TOKEN_RE.match(token):
        return None
    return token.lower()


def _split_blocks(lines: Sequence[str]) -> list[_Block]:
    """Split section body lines into paragraph and fenced-code blocks.

    Admonitions are rendered as blockquotes, so a fence inside one starts with
    ``> ``` ``; the quote prefix is stripped before fence detection, which keeps
    such code atomic instead of letting it be sliced as prose.
    """
    blocks: list[_Block] = []
    prose: list[str] = []

    def flush() -> None:
        if not prose:
            return
        text = "\n".join(prose).strip("\n")
        prose.clear()
        if text.strip():
            blocks.append(_Block(text=text, kind="prose"))

    index = 0
    while index < len(lines):
        content = QUOTE_PREFIX_RE.sub("", lines[index])
        fence = CODE_FENCE_RE.match(content)
        if fence:
            flush()
            marker = fence.group(1)
            language = _language_of(content)
            code_lines = [content]
            in_fence = True
            index += 1
            while index < len(lines):
                current = QUOTE_PREFIX_RE.sub("", lines[index])
                in_fence, marker, is_fence = _advance_fence(
                    current,
                    in_fence=in_fence,
                    fence_marker=marker,
                )
                code_lines.append(current)
                index += 1
                if is_fence and not in_fence:
                    break
            blocks.append(
                _Block(text="\n".join(code_lines), kind="code", language=language)
            )
            continue

        if content.strip():
            prose.append(lines[index])
        else:
            flush()
        index += 1

    flush()
    return blocks


def _hard_split(text: str, budget: int) -> list[str]:
    """Last-resort split for a unit that is longer than the budget alone."""
    limit = max(1, budget * TOKEN_CHARS)
    words = text.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        if current and size + len(word) + 1 > limit:
            pieces.append(" ".join(current))
            current = [word]
            size = len(word)
        else:
            current.append(word)
            size += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_prose(block: _Block, budget: int) -> list[_Block]:
    """Split an over-long paragraph into sentence-sized blocks."""
    pieces: list[_Block] = []
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(block.text)]
    sentences = [sentence for sentence in sentences if sentence]
    for index, sentence in enumerate(sentences):
        # Sentences from one paragraph stay on one line; a blank line marks a
        # real paragraph boundary (the next original prose block).
        separator = "\n\n" if index == 0 else " "
        if count_tokens(sentence) <= budget:
            pieces.append(_Block(text=sentence, kind="prose", separator=separator))
        else:
            for part_index, part in enumerate(_hard_split(sentence, budget)):
                pieces.append(
                    _Block(
                        text=part,
                        kind="prose",
                        separator=separator if part_index == 0 else " ",
                    )
                )
    return pieces or [block]


def _split_long_lines(lines: Sequence[str], limit_chars: int) -> list[str]:
    """Break single lines that are longer than a whole chunk budget.

    Some docs embed multi-kilobyte base64 literals; leaving one intact would
    produce a chunk far over budget, so such a line is cut by characters.
    """
    expanded: list[str] = []
    for line in lines:
        if len(line) <= limit_chars:
            expanded.append(line)
            continue
        expanded.extend(
            line[offset : offset + limit_chars]
            for offset in range(0, len(line), limit_chars)
        )
    return expanded


def _split_code(block: _Block, budget: int) -> list[_Block]:
    """Split an over-long fenced block at blank lines / definitions."""
    lines = block.text.splitlines()
    if len(lines) < 3:
        return [block]

    opening, closing = lines[0], lines[-1]
    limit_chars = max(400, budget * TOKEN_CHARS // 2)
    body = _split_long_lines(lines[1:-1], limit_chars)
    piece_overhead = count_tokens(opening) + count_tokens(closing)
    pieces: list[_Block] = []

    start = 0
    while start < len(body):
        cursor = start
        size = piece_overhead
        split_at: int | None = None

        while cursor < len(body):
            line = body[cursor]
            line_tokens = count_tokens(line) + 1
            if size + line_tokens > budget and cursor > start:
                break
            if DEFINITION_RE.match(line) and cursor > start:
                split_at = cursor
            size += line_tokens
            cursor += 1

        if cursor < len(body) and split_at is not None and split_at > start:
            cursor = split_at

        chunk_lines = body[start:cursor]
        while chunk_lines and not chunk_lines[0].strip():
            chunk_lines.pop(0)
        while chunk_lines and not chunk_lines[-1].strip():
            chunk_lines.pop()
        if chunk_lines:
            pieces.append(
                _Block(
                    text="\n".join([opening, *chunk_lines, closing]),
                    kind="code",
                    language=block.language,
                )
            )
        start = cursor
        while start < len(body) and not body[start].strip():
            start += 1

    return pieces or [block]


def _split_block(block: _Block, budget: int, code_budget: int) -> list[_Block]:
    if block.kind == "code":
        if count_tokens(block.text) <= code_budget:
            return [block]
        return _split_code(block, code_budget)
    if count_tokens(block.text) <= budget:
        return [block]
    return _split_prose(block, budget)


def _pack(blocks: Sequence[_Block], budget: int) -> list[list[_Block]]:
    groups: list[list[_Block]] = []
    current: list[_Block] = []
    size = 0
    for block in blocks:
        tokens = count_tokens(block.text)
        if current and size + tokens > budget:
            groups.append(current)
            current = [block]
            size = tokens
        else:
            current.append(block)
            size += tokens
    if current:
        groups.append(current)
    return groups


def _overlap_prefix(text: str, budget: int) -> str:
    """Return the tail of *text* worth repeating as overlap, within *budget*."""
    units = [line for line in text.splitlines() if line.strip()]
    if len(units) <= 1:
        units = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]

    taken: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = count_tokens(unit) + 1
        if total + unit_tokens > budget:
            break
        taken.insert(0, unit)
        total += unit_tokens
    return "\n".join(taken)


def _overlap_from_prose(blocks: Sequence[_Block], budget: int) -> str:
    """Overlap only the trailing prose run; never carry a code fragment."""
    trailing: list[_Block] = []
    for block in reversed(blocks):
        if block.kind != "prose":
            break
        trailing.append(block)
    if not trailing:
        return ""
    return _overlap_prefix(_join(list(reversed(trailing))), budget)


def _pack_children(
    blocks: Sequence[_Block],
    child_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, list[_Block]]]:
    """Pack blocks into children, returning ``(overlap_prefix, blocks)`` pairs."""
    children: list[tuple[str, list[_Block]]] = []
    index = 0
    prefix = ""

    while index < len(blocks):
        # ``+ 2`` covers the blank line inserted between prefix and body.
        prefix_tokens = count_tokens(prefix) + 2 if prefix else 0
        budget = child_tokens - prefix_tokens
        if prefix and count_tokens(blocks[index].text) > budget:
            # An overlap prefix must never starve the child; drop it instead.
            prefix = ""
            budget = child_tokens
        budget = max(budget, 1)
        group: list[_Block] = []
        size = 0
        while index < len(blocks):
            tokens = count_tokens(blocks[index].text)
            if group and size + tokens > budget:
                break
            group.append(blocks[index])
            size += tokens
            index += 1
        if not group:
            group.append(blocks[index])
            index += 1

        children.append((prefix, group))
        prefix = ""
        if overlap_tokens > 0:
            prefix = _overlap_from_prose(group, overlap_tokens)

    return children


def _section_body(section: Section) -> list[str]:
    lines = list(section.lines)
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return lines


def _section_hash(doc: ParsedDocument, section: Section) -> str:
    key = f"{doc.doc_id}|{'/'.join(section.path)}|{section.anchor or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def _batch_hash(doc: ParsedDocument, units: Sequence[_SectionBlocks]) -> str:
    if len(units) == 1:
        return _section_hash(doc, units[0].section)
    key = "|".join(_section_hash(doc, unit.section) for unit in units)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def _content_type(blocks: Sequence[_Block]) -> ContentType:
    kinds = {block.kind for block in blocks}
    if kinds == {"code"}:
        return "code"
    if "code" in kinds:
        return "mixed"
    return "prose"


def _code_language(blocks: Sequence[_Block]) -> str | None:
    languages = {block.language for block in blocks if block.kind == "code"}
    if len(languages) == 1:
        return next(iter(languages))
    return None


def _join(blocks: Sequence[_Block]) -> str:
    """Join blocks using the separator recorded on each block."""
    parts: list[str] = []
    for index, block in enumerate(blocks):
        parts.append(block.text if index == 0 else f"{block.separator}{block.text}")
    return "".join(parts)


def _header_for_path(path: Sequence[str]) -> str:
    return "[" + " > ".join(["FastAPI", *path]) + "]"


def _header(section: Section) -> str:
    return _header_for_path(section.path)


def _common_path(units: Sequence[_SectionBlocks]) -> list[str]:
    if not units:
        return []
    paths = [unit.section.path for unit in units]
    common: list[str] = []
    for parts in zip(*paths, strict=False):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return common


def _section_text(section: Section, blocks: Sequence[_Block]) -> str:
    heading = f"{'#' * section.heading.level} {section.title}"
    return f"{heading}\n\n{_join(blocks)}"


def _section_units(
    doc: ParsedDocument,
    *,
    child_tokens: int,
    code_max_tokens: int,
    release_notes_latest_versions: int | None,
) -> list[_SectionBlocks]:
    sections = doc.sections
    if doc.doc_type == "release_notes":
        sections = _release_note_sections(sections, release_notes_latest_versions)

    units: list[_SectionBlocks] = []
    for section in sections:
        body = _section_body(section)
        if not body:
            continue
        blocks = [
            part
            for block in _split_blocks(body)
            for part in _split_block(block, child_tokens, code_max_tokens)
        ]
        if not blocks:
            continue
        heading = f"{'#' * section.heading.level} {section.title}"
        tokens = sum(count_tokens(block.text) for block in blocks)
        units.append(
            _SectionBlocks(
                section=section,
                blocks=blocks,
                tokens=tokens + count_tokens(heading) + 1,
            )
        )
    return units


def _release_note_sections(
    sections: Sequence[Section],
    limit: int | None,
) -> list[Section]:
    """Keep the newest N version sections of the release-notes changelog.

    FastAPI's release notes are longer than the rest of the documentation
    combined. Indexing every historical version would swamp normal queries, so
    the normal index keeps the newest versions (``None`` means keep all).
    """
    if limit is None:
        return list(sections)
    versions = [
        section
        for section in sections
        if section.heading.level == 2 and VERSION_HEADING_RE.match(section.title)
    ]
    if len(versions) <= limit:
        return list(sections)

    keep_versions = {id(section) for section in versions[: max(limit, 0)]}
    kept: list[Section] = []
    keep = False
    for section in sections:
        if section.heading.level == 1:
            keep = True
        elif section.heading.level == 2:
            keep = (
                id(section) in keep_versions
                or section.title.strip() == LATEST_CHANGES_TITLE
            )
        if keep:
            kept.append(section)
    return kept


def _batch_units(
    units: Sequence[_SectionBlocks],
    parent_tokens: int,
) -> list[list[_SectionBlocks]]:
    """Greedily pack consecutive whole sections into parent-sized batches."""
    batches: list[list[_SectionBlocks]] = []
    current: list[_SectionBlocks] = []
    size = 0
    for unit in units:
        if unit.tokens > parent_tokens:
            if current:
                batches.append(current)
                current = []
                size = 0
            batches.append([unit])
            continue
        if current and size + unit.tokens > parent_tokens:
            batches.append(current)
            current = []
            size = 0
        current.append(unit)
        size += unit.tokens
    if current:
        batches.append(current)
    return batches


def _make_parent(
    doc: ParsedDocument,
    units: Sequence[_SectionBlocks],
    blocks: Sequence[_Block],
    *,
    text: str,
    parent_id: str,
    parent_index: int,
) -> Chunk:
    section_path = _common_path(units)
    single = units[0].section if len(units) == 1 else None
    return Chunk(
        chunk_id=parent_id,
        doc_id=doc.doc_id,
        parent_id=parent_id,
        level="parent",
        text=text,
        embed_text=f"{_header_for_path(section_path)}\n{text}",
        token_count=count_tokens(text),
        section_path=section_path,
        anchor=single.anchor if single else None,
        url=(
            f"{doc.url}#{single.anchor}"
            if single and single.anchor
            else doc.url
        ),
        content_type=_content_type(blocks),
        code_language=_code_language(blocks),
        is_deprecated=any(unit.section.is_deprecated for unit in units),
        deprecation_hint=any(unit.section.deprecation_hint for unit in units),
        section_hash=_batch_hash(doc, units),
        chunk_index=parent_index,
    )


def _make_child(
    doc: ParsedDocument,
    unit: _SectionBlocks,
    *,
    parent_id: str,
    child_index: int,
    prefix: str,
    blocks: Sequence[_Block],
) -> Chunk:
    section = unit.section
    body_text = _join(blocks)
    child_text = f"{prefix}\n\n{body_text}" if prefix else body_text
    typed_blocks = (
        [_Block(text=prefix, kind="prose"), *blocks] if prefix else list(blocks)
    )
    return Chunk(
        chunk_id=f"{doc.doc_id}_{_section_hash(doc, section)}_{child_index}",
        doc_id=doc.doc_id,
        parent_id=parent_id,
        level="child",
        text=child_text,
        embed_text=f"{_header(section)}\n{child_text}",
        token_count=count_tokens(child_text),
        section_path=list(section.path),
        anchor=section.anchor,
        url=f"{doc.url}#{section.anchor}" if section.anchor else doc.url,
        content_type=_content_type(typed_blocks),
        code_language=_code_language(typed_blocks),
        is_deprecated=section.is_deprecated,
        deprecation_hint=section.deprecation_hint,
        section_hash=_section_hash(doc, section),
        chunk_index=child_index,
    )


def chunk_document(
    doc: ParsedDocument,
    *,
    parent_tokens: int = 1400,
    child_tokens: int = 400,
    overlap_tokens: int = 64,
    code_max_tokens: int = CODE_BLOCK_MAX_TOKENS,
    release_notes_latest_versions: int | None = 20,
) -> list[Chunk]:
    """Chunk one parsed document into parents followed by their children."""
    if child_tokens <= 0 or parent_tokens < child_tokens:
        raise ValueError("expected 0 < child_tokens <= parent_tokens")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")

    units = _section_units(
        doc,
        child_tokens=child_tokens,
        code_max_tokens=code_max_tokens,
        release_notes_latest_versions=release_notes_latest_versions,
    )
    if not units:
        return []

    chunks: list[Chunk] = []
    parent_index = 0
    for batch in _batch_units(units, parent_tokens):
        # A single oversized section is packed into several parents, then its
        # children stay within those parent groups.
        if len(batch) == 1 and batch[0].tokens > parent_tokens:
            unit = batch[0]
            child_index = 0
            for parent_blocks in _pack(unit.blocks, parent_tokens):
                parent_id = f"{doc.doc_id}_p{parent_index}"
                parent_text = _section_text(unit.section, parent_blocks)
                chunks.append(
                    _make_parent(
                        doc,
                        batch,
                        parent_blocks,
                        text=parent_text,
                        parent_id=parent_id,
                        parent_index=parent_index,
                    )
                )
                for prefix, child_blocks in _pack_children(
                    parent_blocks, child_tokens, overlap_tokens
                ):
                    chunks.append(
                        _make_child(
                            doc,
                            unit,
                            parent_id=parent_id,
                            child_index=child_index,
                            prefix=prefix,
                            blocks=child_blocks,
                        )
                    )
                    child_index += 1
                parent_index += 1
            continue

        parent_id = f"{doc.doc_id}_p{parent_index}"
        all_blocks = [block for unit in batch for block in unit.blocks]
        parent_text = "\n\n".join(
            _section_text(unit.section, unit.blocks) for unit in batch
        )
        chunks.append(
            _make_parent(
                doc,
                batch,
                all_blocks,
                text=parent_text,
                parent_id=parent_id,
                parent_index=parent_index,
            )
        )
        for unit in batch:
            child_index = 0
            for prefix, child_blocks in _pack_children(
                unit.blocks, child_tokens, overlap_tokens
            ):
                chunks.append(
                    _make_child(
                        doc,
                        unit,
                        parent_id=parent_id,
                        child_index=child_index,
                        prefix=prefix,
                        blocks=child_blocks,
                    )
                )
                child_index += 1
        parent_index += 1

    return chunks


def chunk_documents(
    documents: Iterable[ParsedDocument],
    *,
    parent_tokens: int = 1400,
    child_tokens: int = 400,
    overlap_tokens: int = 64,
    code_max_tokens: int = CODE_BLOCK_MAX_TOKENS,
    release_notes_latest_versions: int | None = 20,
) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(
            chunk_document(
                doc,
                parent_tokens=parent_tokens,
                child_tokens=child_tokens,
                overlap_tokens=overlap_tokens,
                code_max_tokens=code_max_tokens,
                release_notes_latest_versions=release_notes_latest_versions,
            )
        )
    return chunks
