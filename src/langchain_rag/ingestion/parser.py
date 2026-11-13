"""Parse FastAPI Markdown documentation.

The parser resolves ``{* ... *}`` code includes, preserves admonitions as
blockquotes, separates heading anchors, and tracks code fences so that
``#`` comments inside code are never mistaken for headings.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

SITE_URL = "https://fastapi.tiangolo.com"

CODE_INCLUDE_RE = re.compile(r"^\{\*\s*(\S+)\s*(.*?)\s*\}\s*$")
LINE_RANGE_RE = re.compile(r"ln\[(\d+)(?::(\d+))?\]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s*\{\s*#([^}\s]+)\s*\})?\s*$")
ADMONITION_START_RE = re.compile(
    r"^(\s*)(/{3,})\s*([A-Za-z0-9_-]+)(?:\s*\|\s*(.+?))?\s*$"
)
ADMONITION_END_RE = re.compile(r"^(\s*)(/{3,})\s*$")
CODE_FENCE_RE = re.compile(r"^\s*(`{3,4})(.*)$")

# ``deprecat`` covers deprecated / deprecate / deprecating / deprecation.
DEPRECATION_RE = re.compile(r"\bdeprecat", re.IGNORECASE)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
BLOCKQUOTE_PREFIX_RE = re.compile(r"^(?:>\s?)+")
# Titles like "Deprecate a path operation" document a feature; they do not
# describe an API that is deprecated.
INSTRUCTIONAL_DEPRECATION_RE = re.compile(
    r"^\s*(?:documentation\s+)?deprecat(?:e|ing)\b",
    re.IGNORECASE,
)
DEPRECATED_TITLE_RE = re.compile(r"^\s*deprecated\b", re.IGNORECASE)
PARAMETER_RE = re.compile(
    r"\b(?:parameters?|arguments?|keywords?|query parameters?|path parameters?)\b",
    re.IGNORECASE,
)
DEPRECATED_PHRASE_RE = re.compile(
    r"\b(?:is|are|was|were|has been|have been)\s+(?:now\s+)?deprecated\b",
    re.IGNORECASE,
)

DOC_TYPE_DIRS = {"tutorial", "advanced", "how-to", "deployment", "learn", "reference"}
REPO_ANCHORS = ("docs_src", "fastapi", "scripts", "tests")
META_FILES = {
    "about",
    "contributing.md",
    "external-links.md",
    "fastapi-people.md",
    "help-fastapi.md",
    "management.md",
    "newsletter.md",
    "resources",
    "translation-banner.md",
    "translations.md",
}
SKIP_FILES = {"_llm-test.md"}

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".toml": "toml",
    ".md": "markdown",
    ".txt": "text",
    ".html": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
}


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    anchor: str | None = None


@dataclass
class Section:
    heading: Heading
    path: list[str]
    lines: list[str] = field(default_factory=list)
    deprecated_admonitions: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.heading.title

    @property
    def anchor(self) -> str | None:
        return self.heading.anchor

    @property
    def is_deprecated(self) -> bool:
        """Strong signal: the section itself documents a deprecated API."""
        return _heading_is_deprecated(self.heading.title) or bool(
            self.deprecated_admonitions
        )

    @property
    def mentions_deprecation(self) -> bool:
        """Weak signal: any prose (not only headings/admonitions) mentions it."""
        return self.is_deprecated or mentions_deprecation(self.lines)

    @property
    def deprecation_hint(self) -> bool:
        """Weak signal that is not strong enough to mark the section deprecated."""
        return self.mentions_deprecation and not self.is_deprecated


@dataclass
class ParsedDocument:
    source_path: Path
    relative_path: Path
    url: str
    title: str
    doc_type: str
    markdown: str
    sections: list[Section]
    code_includes: list[str]
    content_sha256: str
    frontmatter: dict[str, object] = field(default_factory=dict)
    preamble_deprecated_admonitions: list[str] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:16]

    @property
    def has_deprecated_section(self) -> bool:
        """True when a heading/admonition marks a deprecated API (strong)."""
        return any(section.is_deprecated for section in self.sections)

    @property
    def has_deprecation_notes(self) -> bool:
        """True when deprecation is mentioned anywhere in prose (weak)."""
        return bool(self.preamble_deprecated_admonitions) or any(
            section.mentions_deprecation for section in self.sections
        )

    def metadata(
        self,
        *,
        docs_version: str,
        source_ref: str,
        fetched_at: str,
        index_build_id: str,
    ) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "canonical_url": self.url,
            "title": self.title,
            "section_path": [],
            "product": "fastapi",
            "language": "python",
            "doc_type": self.doc_type,
            "docs_version": docs_version,
            "source_ref": source_ref,
            "fetched_at": fetched_at,
            "is_deprecated": False,
            "has_deprecated_section": self.has_deprecated_section,
            "has_deprecation_notes": self.has_deprecation_notes,
            "content_sha256": self.content_sha256,
            "index_build_id": index_build_id,
        }


def parse_heading(line: str) -> Heading | None:
    """Return a Heading when *line* is an ATX heading outside code."""
    match = HEADING_RE.match(line)
    if not match:
        return None
    return Heading(
        level=len(match.group(1)),
        title=match.group(2).strip(),
        anchor=match.group(3),
    )


def guess_doc_type(relative_path: Path) -> str:
    """Classify a document from its path relative to ``docs/en/docs``."""
    parts = relative_path.parts
    if parts and parts[0] in DOC_TYPE_DIRS:
        return parts[0]
    if relative_path.name == "release-notes.md":
        return "release_notes"
    if relative_path.name in META_FILES or (parts and parts[0] in META_FILES):
        return "meta"
    return "concept"


def url_for(relative_path: Path, site_url: str = SITE_URL) -> str:
    """Map a docs-relative Markdown path to its public URL."""
    base = site_url.rstrip("/")
    if relative_path.name == "index.md":
        parent = relative_path.parent.as_posix()
        return f"{base}/{parent}/" if parent != "." else f"{base}/"
    return f"{base}/{relative_path.with_suffix('').as_posix()}/"


def resolve_include_path(
    include: str,
    *,
    doc_path: Path,
    docs_root: Path,
    repo_root: Path,
) -> Path:
    """Resolve a FastAPI code include to a path in the repository.

    Includes look like ``../../docs_src/tutorial/first_steps.py``. They are
    written for the built site, where the docs root has the repository root as
    an ancestor, so the reliable anchor is the ``docs_src`` path component.
    """
    include_path = Path(include.replace("\\", "/"))
    parts = include_path.parts
    for anchor in REPO_ANCHORS:
        if anchor in parts:
            return repo_root.joinpath(*parts[parts.index(anchor) :])

    candidates = [
        doc_path.parent / include_path,
        docs_root / include_path,
        repo_root / include_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _fence_for(content: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _advance_fence(
    line: str,
    *,
    in_fence: bool,
    fence_marker: str,
) -> tuple[bool, str, bool]:
    """Update code-fence state for one line.

    Returns ``(in_fence, fence_marker, is_fence_line)``. A closing fence must
    carry no info string and be at least as long as the opening fence.
    """
    match = CODE_FENCE_RE.match(line)
    if not match:
        return in_fence, fence_marker, False

    marker = match.group(1)
    rest = match.group(2).strip()
    if not in_fence:
        return True, marker, True
    if not rest and len(marker) >= len(fence_marker):
        return False, "", True
    return in_fence, fence_marker, True


def _language_for(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


def expand_code_includes(
    lines: list[str],
    *,
    doc_path: Path,
    docs_root: Path,
    repo_root: Path,
) -> tuple[list[str], list[str]]:
    """Expand code include lines into fenced code blocks.

    Returns the expanded lines and the repository-relative paths that were
    included. Missing files become an HTML comment instead of aborting the
    whole page, so one broken include does not drop an entire document.
    """
    expanded: list[str] = []
    included: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in lines:
        in_fence, fence_marker, is_fence = _advance_fence(
            line,
            in_fence=in_fence,
            fence_marker=fence_marker,
        )
        if is_fence or in_fence:
            expanded.append(line)
            continue

        match = CODE_INCLUDE_RE.match(line)
        if not match:
            expanded.append(line)
            continue

        include = match.group(1)
        resolved = resolve_include_path(
            include,
            doc_path=doc_path,
            docs_root=docs_root,
            repo_root=repo_root,
        )
        if not resolved.exists():
            expanded.append(f"<!-- missing code include: {include} -->")
            continue

        content = resolved.read_text(encoding="utf-8").rstrip("\n")
        line_range = LINE_RANGE_RE.search(match.group(2))
        if line_range:
            start = int(line_range.group(1))
            end = int(line_range.group(2) or start)
            content = "\n".join(content.splitlines()[max(start - 1, 0) : end])
        fence = _fence_for(content)
        language = _language_for(resolved)
        expanded.extend([f"{fence}{language}", content, fence])
        try:
            included.append(resolved.relative_to(repo_root).as_posix())
        except ValueError:
            included.append(resolved.as_posix())

    return expanded, included


def _parse_frontmatter(lines: list[str]) -> tuple[dict[str, object], list[str]]:
    if not lines or lines[0].strip() != "---":
        return {}, lines

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            try:
                import yaml

                data = yaml.safe_load("\n".join(lines[1:index])) or {}
            except Exception:  # noqa: BLE001 - malformed frontmatter is data, not fatal
                return {}, lines
            return data if isinstance(data, dict) else {}, lines[index + 1 :]
    return {}, lines


def _render_admonition(
    kind: str,
    body: list[str],
    title: str | None = None,
    indent: str = "",
) -> list[str]:
    rendered = [f"{indent}> **{kind.upper()}**"]
    if title:
        rendered.append(f"{indent}> *{title}*")
    rendered.append(f"{indent}>")
    for line in body:
        rendered.append(f"{indent}> {line}" if line else f"{indent}>")
    return rendered


def _parse_admonition(
    lines: list[str],
    start: int,
    opener: re.Match[str],
    *,
    depth: int = 0,
) -> tuple[list[str], int, bool]:
    """Render one admonition block.

    Returns ``(rendered_lines, next_index, is_deprecated)``. MkDocs
    Material lets a ``////`` tab group contain ``///`` notes and allows
    admonitions indented inside list items, so both the slash count and the
    leading indent are preserved.
    """
    indent = opener.group(1)
    fence_length = len(opener.group(2))
    kind = opener.group(3)
    title = opener.group(4)
    strong = False
    body: list[str] = []
    cursor = start + 1

    while cursor < len(lines):
        line = lines[cursor]
        nested = ADMONITION_START_RE.match(line)
        if nested and depth < 3:
            nested_rendered, cursor, nested_deprecated = _parse_admonition(
                lines, cursor, nested, depth=depth + 1
            )
            strong = strong or nested_deprecated
            body.extend(nested_rendered)
            continue

        end = ADMONITION_END_RE.match(line)
        if end and len(end.group(2)) == fence_length:
            cursor += 1
            break

        body.append(line[len(indent) :] if indent and line.startswith(indent) else line)
        cursor += 1

    strong = strong or _admonition_is_deprecated(title, body)
    return _render_admonition(kind, body, title, indent), cursor, strong


def _heading_is_deprecated(title: str) -> bool:
    """Return True when a heading explicitly marks its API as deprecated.

    "Deprecate a path operation" and "Deprecating parameters" are guides for
    marking something else as deprecated, not deprecated APIs themselves.
    """
    if INSTRUCTIONAL_DEPRECATION_RE.search(title):
        return False
    if PARAMETER_RE.search(title):
        return False
    return bool(DEPRECATION_RE.search(title))


def _plain_prose(lines: list[str]) -> str:
    """Return non-code prose with inline code spans blanked."""
    kept: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in lines:
        content = BLOCKQUOTE_PREFIX_RE.sub("", line)
        in_fence, fence_marker, is_fence = _advance_fence(
            content,
            in_fence=in_fence,
            fence_marker=fence_marker,
        )
        if is_fence or in_fence:
            continue
        kept.append(INLINE_CODE_RE.sub(" ", content))
    return "\n".join(kept)


def _admonition_is_deprecated(title: str | None, body: list[str]) -> bool:
    """Return True only for a section-level deprecation notice.

    Parameter/keyword deprecations are weak hints: they say one input is
    deprecated, not that the whole section's API is gone.
    """
    title_text = title or ""
    prose = _plain_prose(body)
    combined = f"{title_text}\n{prose}"
    if not DEPRECATION_RE.search(combined):
        return False
    if PARAMETER_RE.search(combined):
        return False
    if DEPRECATED_TITLE_RE.search(title_text):
        return True
    return bool(DEPRECATED_PHRASE_RE.search(prose))


def mentions_deprecation(lines: list[str]) -> bool:
    """Return True when non-code prose mentions deprecation.

    Fenced code is skipped (a ``# deprecated`` comment is not a notice) and
    inline code spans are blanked, so a parameter literally named
    ``deprecated`` does not count as a notice.
    """
    return bool(DEPRECATION_RE.search(_plain_prose(lines)))


def parse_markdown(
    text: str,
    *,
    doc_path: Path,
    docs_root: Path,
    repo_root: Path,
    site_url: str = SITE_URL,
) -> ParsedDocument:
    """Parse one FastAPI Markdown document."""
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    frontmatter, lines = _parse_frontmatter(raw_lines)
    lines, code_includes = expand_code_includes(
        lines,
        doc_path=doc_path,
        docs_root=docs_root,
        repo_root=repo_root,
    )

    preamble: list[str] = []
    preamble_deprecated: list[str] = []
    sections: list[Section] = []
    stack: list[Heading] = []
    current: Section | None = None
    in_fence = False
    fence_marker = ""

    index = 0
    while index < len(lines):
        line = lines[index]

        in_fence, fence_marker, is_fence = _advance_fence(
            line,
            in_fence=in_fence,
            fence_marker=fence_marker,
        )
        if is_fence:
            (current.lines if current else preamble).append(line)
            index += 1
            continue

        if not in_fence:
            admonition = ADMONITION_START_RE.match(line)
            if admonition:
                kind = admonition.group(3)
                rendered, next_index, deprecated = _parse_admonition(
                    lines, index, admonition
                )
                if current is not None:
                    current.lines.extend(rendered)
                    if deprecated:
                        current.deprecated_admonitions.append(kind)
                else:
                    preamble.extend(rendered)
                    if deprecated:
                        preamble_deprecated.append(kind)
                index = next_index
                continue

            heading = parse_heading(line)
            if heading:
                while stack and stack[-1].level >= heading.level:
                    stack.pop()
                stack.append(heading)
                current = Section(
                    heading=heading,
                    path=[item.title for item in stack],
                )
                sections.append(current)
                current.lines.append(f"{'#' * heading.level} {heading.title}")
                index += 1
                continue

        (current.lines if current else preamble).append(line)
        index += 1

    relative_path = doc_path.relative_to(docs_root)
    title = next(
        (section.title for section in sections if section.heading.level == 1),
        doc_path.stem,
    )
    markdown_lines = preamble + [
        line for section in sections for line in section.lines
    ]
    markdown = "\n".join(markdown_lines).strip() + "\n"
    normalized = "\n".join(line.rstrip() for line in markdown.splitlines())
    content_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return ParsedDocument(
        source_path=doc_path,
        relative_path=relative_path,
        url=url_for(relative_path, site_url=site_url),
        title=title,
        doc_type=guess_doc_type(relative_path),
        markdown=markdown,
        sections=sections,
        code_includes=code_includes,
        content_sha256=content_sha256,
        frontmatter=frontmatter,
        preamble_deprecated_admonitions=preamble_deprecated,
    )


def parse_file(
    doc_path: Path,
    *,
    docs_root: Path,
    repo_root: Path,
    site_url: str = SITE_URL,
) -> ParsedDocument:
    return parse_markdown(
        doc_path.read_text(encoding="utf-8"),
        doc_path=doc_path,
        docs_root=docs_root,
        repo_root=repo_root,
        site_url=site_url,
    )


def iter_documents(
    *,
    docs_root: Path,
    repo_root: Path,
    site_url: str = SITE_URL,
) -> Iterator[ParsedDocument]:
    """Yield parsed documents under *docs_root*, skipping test scaffolding."""
    for doc_path in sorted(docs_root.rglob("*.md")):
        if doc_path.name in SKIP_FILES:
            continue
        yield parse_file(
            doc_path,
            docs_root=docs_root,
            repo_root=repo_root,
            site_url=site_url,
        )
