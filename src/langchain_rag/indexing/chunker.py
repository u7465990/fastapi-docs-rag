"""Split parsed FastAPI documents into parent and child chunks.

A parent keeps one whole section (up to ~1400 tokens) for the LLM; children
(~400 tokens) are what gets embedded. Code fences are atomic: a fence is never
cut mid-block, and only a block larger than 800 tokens is split on blank lines
or definition boundaries, with the language marker kept on every piece.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_rag.ingestion.parser import (
    CODE_FENCE_RE,
    ParsedDocument,
    Section,
    _advance_fence,
)

#: English prose averages ~4 characters per token. This deliberately
#: over-estimates code, which keeps the budget on the safe side.
TOKEN_CHARS = 4

#: A code block up to this size stays atomic even if it exceeds the child
#: budget: cutting code is worse than a slightly oversized chunk.
CODE_BLOCK_MAX_TOKENS = 800

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DEFINITION_RE = re.compile(r"^(?:async\s+)?(?:def|class)\s")
QUOTE_PREFIX_RE = re.compile(r"^\s*(?:>\s?)+")

BlockKind = Literal["prose", "code"]
ContentType = Literal["prose", "code", "mixed"]
ChunkLevel = Literal["parent", "child"]


@dataclass(frozen=True)
class _Block:
    """An atomic unit of text: a paragraph or a whole fenced code block."""

    text: str
    kind: BlockKind = "prose"
    language: str | None = None


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
    for sentence in SENTENCE_RE.split(block.text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if count_tokens(sentence) <= budget:
            pieces.append(_Block(text=sentence, kind="prose"))
        else:
            pieces.extend(
                _Block(text=part, kind="prose")
                for part in _hard_split(sentence, budget)
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
        if overlap_tokens > 0 and group[-1].kind == "prose":
            prefix = _overlap_prefix(_join(group), overlap_tokens)

    return children


def _section_body(section: Section) -> list[str]:
    lines = list(section.lines)
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return lines


def _section_hash(doc: ParsedDocument, section: Section) -> str:
    key = f"{doc.doc_id}|{'/'.join(section.path)}|{section.anchor or ''}"
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
    """Join blocks: blank lines around code, single newlines between prose."""
    parts: list[str] = []
    for index, block in enumerate(blocks):
        if index == 0:
            parts.append(block.text)
            continue
        previous = blocks[index - 1]
        separator = "\n\n" if "code" in (previous.kind, block.kind) else "\n"
        parts.append(separator + block.text)
    return "".join(parts)


def _header(section: Section) -> str:
    return "[" + " > ".join(["FastAPI", *section.path]) + "]"


def chunk_document(
    doc: ParsedDocument,
    *,
    parent_tokens: int = 1400,
    child_tokens: int = 400,
    overlap_tokens: int = 64,
    code_max_tokens: int = CODE_BLOCK_MAX_TOKENS,
) -> list[Chunk]:
    """Chunk one parsed document into parents followed by their children."""
    if child_tokens <= 0 or parent_tokens < child_tokens:
        raise ValueError("expected 0 < child_tokens <= parent_tokens")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")

    chunks: list[Chunk] = []
    for section in doc.sections:
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

        section_hash = _section_hash(doc, section)
        header = _header(section)
        url = f"{doc.url}#{section.anchor}" if section.anchor else doc.url
        child_index = 0

        for parent_index, parent_blocks in enumerate(_pack(blocks, parent_tokens)):
            parent_text = _join(parent_blocks)
            parent_id = f"{doc.doc_id}_{section_hash}_p{parent_index}"
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_id=doc.doc_id,
                    parent_id=parent_id,
                    level="parent",
                    text=parent_text,
                    embed_text=f"{header}\n{parent_text}",
                    token_count=count_tokens(parent_text),
                    section_path=list(section.path),
                    anchor=section.anchor,
                    url=url,
                    content_type=_content_type(parent_blocks),
                    code_language=_code_language(parent_blocks),
                    is_deprecated=section.is_deprecated,
                    section_hash=section_hash,
                    chunk_index=parent_index,
                )
            )

            for prefix, child_blocks in _pack_children(
                parent_blocks,
                child_tokens,
                overlap_tokens,
            ):
                body_text = _join(child_blocks)
                child_text = f"{prefix}\n\n{body_text}" if prefix else body_text
                typed_blocks = (
                    [_Block(text=prefix, kind="prose"), *child_blocks]
                    if prefix
                    else child_blocks
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_{section_hash}_{child_index}",
                        doc_id=doc.doc_id,
                        parent_id=parent_id,
                        level="child",
                        text=child_text,
                        embed_text=f"{header}\n{child_text}",
                        token_count=count_tokens(child_text),
                        section_path=list(section.path),
                        anchor=section.anchor,
                        url=url,
                        content_type=_content_type(typed_blocks),
                        code_language=_code_language(typed_blocks),
                        is_deprecated=section.is_deprecated,
                        section_hash=section_hash,
                        chunk_index=child_index,
                    )
                )
                child_index += 1

    return chunks


def chunk_documents(
    documents: Iterable[ParsedDocument],
    *,
    parent_tokens: int = 1400,
    child_tokens: int = 400,
    overlap_tokens: int = 64,
    code_max_tokens: int = CODE_BLOCK_MAX_TOKENS,
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
            )
        )
    return chunks
