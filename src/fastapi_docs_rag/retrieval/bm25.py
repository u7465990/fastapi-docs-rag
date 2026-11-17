"""Small in-memory BM25 index for the hybrid retrieval baseline."""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from qdrant_client import QdrantClient

from fastapi_docs_rag.retrieval.dense import SearchHit, hit_from_point

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# A deliberately small domain glossary. Chinese queries cannot match English
# BM25 terms without translation; this covers common FastAPI wording and keeps
# the branch deterministic and offline. Unknown Chinese terms still rely on
# dense retrieval.
QUERY_GLOSSARY = {
    "第一个应用": "first steps application",
    "路径参数": "path parameters",
    "查询参数": "query parameters",
    "请求体": "request body",
    "响应模型": "response model",
    "依赖注入": "dependency injection dependencies",
    "依赖": "dependencies",
    "生命周期": "lifespan events",
    "启动": "startup",
    "关闭": "shutdown",
    "后台任务": "background tasks",
    "测试": "testing testclient",
    "错误": "errors exception",
    "异常": "exceptions",
    "中间件": "middleware",
    "跨域": "cors",
    "文件上传": "file upload",
    "表单": "form data",
    "数据库": "sql databases",
    "认证": "authentication",
    "授权": "authorization",
    "密码": "password",
    "令牌": "token",
    "部署": "deployment",
    "多进程": "multiple processes workers",
    "工作进程": "workers",
    "配置": "settings configuration",
    "版本": "release notes version",
    "废弃": "deprecated",
    "迁移": "migrate",
    "校验": "validation",
    "分页": "pagination",
    "上传": "upload",
    "下载": "download",
    "路由": "router",
    "应用": "application",
    "接口": "endpoint",
    "响应": "response",
    "请求": "request",
    "状态码": "status code",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        for part in raw.split("_"):
            if not part:
                continue
            tokens.append(part.lower())
            for camel in CAMEL_RE.split(part):
                if camel and camel.lower() != part.lower():
                    tokens.append(camel.lower())
    return tokens


def expand_query(query: str) -> str:
    # A Chinese-only query has no BM25 terms that can match English documents.
    # Do not inject a noisy glossary translation; let dense retrieval handle it.
    if not TOKEN_RE.search(query):
        return query
    additions: list[str] = []
    for chinese, english in QUERY_GLOSSARY.items():
        if chinese in query:
            additions.append(english)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


@dataclass
class _IndexedDoc:
    hit: SearchHit
    token_counts: Counter[str]
    length: int


class BM25Index:
    def __init__(self, docs: list[SearchHit], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[_IndexedDoc] = []
        for hit in docs:
            text = " ".join(
                [
                    hit.title,
                    " ".join(hit.section_path),
                    hit.text,
                    hit.parent_text,
                ]
            )
            counts = Counter(tokenize(text))
            if counts:
                self.docs.append(
                    _IndexedDoc(hit=hit, token_counts=counts, length=sum(counts.values()))
                )
        self.avg_length = (
            sum(doc.length for doc in self.docs) / len(self.docs) if self.docs else 0.0
        )
        self.document_frequency: Counter[str] = Counter()
        for doc in self.docs:
            self.document_frequency.update(doc.token_counts.keys())

    @classmethod
    def from_qdrant(cls, client: QdrantClient, collection: str) -> BM25Index:
        points: list[Any] = []
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                break
        return cls([hit_from_point(point) for point in points])

    def _matches(
        self,
        hit: SearchHit,
        *,
        doc_type: str | None,
        docs_version: str | None,
        include_deprecated: bool,
        include_meta: bool,
        include_reference: bool,
    ) -> bool:
        if doc_type and hit.doc_type != doc_type:
            return False
        if docs_version and hit.docs_version != docs_version:
            return False
        if not include_deprecated and hit.is_deprecated:
            return False
        if not include_meta and hit.doc_type == "meta":
            return False
        if not include_reference and hit.doc_type == "reference":
            return False
        return True

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        doc_type: str | None = None,
        docs_version: str | None = None,
        include_deprecated: bool = True,
        include_meta: bool = False,
        include_reference: bool = False,
    ) -> list[SearchHit]:
        if not query.strip() or limit <= 0 or not self.docs:
            return []
        query_terms = tokenize(expand_query(query))
        if not query_terms:
            return []
        scores: list[tuple[float, int]] = []
        total = len(self.docs)
        for index, doc in enumerate(self.docs):
            if not self._matches(
                doc.hit,
                doc_type=doc_type,
                docs_version=docs_version,
                include_deprecated=include_deprecated,
                include_meta=include_meta,
                include_reference=include_reference,
            ):
                continue
            score = 0.0
            for term in query_terms:
                tf = doc.token_counts.get(term, 0)
                if not tf:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc.length / (self.avg_length or 1.0)
                )
                score += idf * tf * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((score, index))
        best = heapq.nlargest(limit, scores, key=lambda item: item[0])
        return [
            replace(self.docs[index].hit, score=score)
            for score, index in best
        ]
