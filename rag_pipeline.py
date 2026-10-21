"""Hybrid retrieval over table chunks: BM25 + dense vectors + reciprocal rank fusion + cross-encoder rerank.

- Sparse: rank_bm25 over finance-aware tokens (keeps "mar-26", "p&l", "s&m" intact) for exact matches on
  line items and fiscal periods.
- Dense: BAAI/bge-large-en-v1.5 embeddings in ChromaDB (default) or Qdrant (configs/rag.yaml).
- Fusion: reciprocal rank fusion of both candidate lists.
- Rerank: BAAI/bge-reranker-large cross-encoder picks the final top_k.

Usage:
  python rag_pipeline.py sample_startup_mis.xlsx "What is CM2 for Mar-26?"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common.schemas import TableChunk

ROOT = Path(__file__).resolve().parent
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-&/][a-z0-9]+)*")
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
_MONTH_NAME_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?[\s,'-]*(?:20)?(\d{2})\b", re.IGNORECASE)

# Retrieval-side domain knowledge: which line items each metric needs.
QUERY_EXPANSION = {
    "net revenue": "gross revenue discounts returns",
    "cm1": "net revenue cogs cost of revenue gross revenue discounts returns",
    "cm2": "variable costs fulfillment logistics payment gateway packaging cogs revenue",
    "cm3": "performance marketing variable costs cogs revenue",
    "contribution": "variable costs fulfillment payment gateway packaging cogs revenue",
    "gross margin": "cogs cost of revenue net revenue gross margin %",
    "ebitda": "ebit depreciation amortization net income interest tax one-off items",
    "ltv": "arpu churn gross margin unit economics customers",
    "cac": "new customers s&m spend sales marketing spend unit economics",
    "payback": "cac arpu gross margin new customers s&m spend",
    "break-even": "price per unit variable cost per unit fixed costs pricing",
    "break even": "price per unit variable cost per unit fixed costs pricing",
    "runway": "cash balance cash in cash out burn",
    "burn": "cash in cash out cash balance",
}


def normalize_periods(text: str) -> str:
    """'March 2026' / 'Mar 26' -> 'Mar-26' so BM25 matches table headers."""
    return _MONTH_NAME_RE.sub(lambda m: f"{m.group(1).title()}-{m.group(2)}", text)


def tokenize(text: str) -> list[str]:
    toks = []
    for t in _TOKEN_RE.findall(text.lower()):
        toks.append(t)
        if any(ch in t for ch in "-&/"):
            toks += [p for p in re.split(r"[-&/]", t) if p]
    return toks


def chunk_summary(c: TableChunk) -> str:
    """What the dense model and reranker see: the table's meaning without its digits.

    12 columns of numbers drown semantic models; titles, row labels and periods carry the signal.
    The LLM still receives the full chunk text.
    """
    m = c.metadata
    periods = f"{m.periods[0]} to {m.periods[-1]}" if len(m.periods) > 1 else (m.periods[0] if m.periods else "")
    return (f"{c.title}. Sheet: {m.sheet}. Section: {m.section}. Line items: {', '.join(m.row_labels)}. "
            f"Periods: {periods}. Units: {m.units}.")


def expand_query(q: str) -> str:
    ql = q.lower()
    extra = [v for k, v in QUERY_EXPANSION.items() if k in ql]
    return " ".join([q, *extra])


@dataclass
class Hit:
    chunk: TableChunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    fused: float


class _ChromaStore:
    def __init__(self, persist_dir: str | None):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        self.name = "finance_chunks"

    def reset(self, dim: int) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:  # noqa: BLE001 - collection did not exist
            pass
        # vectors are L2-normalised, so L2 distance ranks the same as cosine similarity
        self.col = self.client.create_collection(self.name, embedding_function=None)

    def add(self, ids: list[str], vecs: list[list[float]]) -> None:
        self.col.add(ids=ids, embeddings=vecs)

    def query(self, vec: list[float], n: int) -> list[str]:
        res = self.col.query(query_embeddings=[vec], n_results=min(n, self.col.count()))
        return res["ids"][0]


class _QdrantStore:
    def __init__(self, persist_dir: str | None):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=persist_dir) if persist_dir else QdrantClient(":memory:")
        self.name = "finance_chunks"
        self.ids: list[str] = []

    def reset(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.name):
            self.client.delete_collection(self.name)
        self.client.create_collection(self.name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        self.ids = []

    def add(self, ids: list[str], vecs: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        base = len(self.ids)
        self.ids += ids
        self.client.upsert(self.name, points=[PointStruct(id=base + i, vector=v) for i, v in enumerate(vecs)])

    def query(self, vec: list[float], n: int) -> list[str]:
        res = self.client.query_points(self.name, query=vec, limit=n)
        return [self.ids[p.id] for p in res.points]


class HybridRetriever:
    def __init__(self, cfg: dict[str, Any] | None = None, load_models: bool = True):
        self.cfg = cfg or yaml.safe_load((ROOT / "configs" / "rag.yaml").read_text(encoding="utf-8"))
        self.chunks: dict[str, TableChunk] = {}
        self.order: list[str] = []
        self.bm25 = None
        self.embedder = self.reranker = self.store = None
        if load_models:  # load_models=False -> BM25 only (tests, offline use)
            from sentence_transformers import CrossEncoder, SentenceTransformer

            dev = self.cfg["device"]
            self.embedder = SentenceTransformer(self.cfg["embedding_model"], device=dev)
            self.reranker = CrossEncoder(self.cfg["reranker_model"], device=dev)
            persist = self.cfg.get("persist_dir")
            self.store = _QdrantStore(persist) if self.cfg["vector_store"] == "qdrant" else _ChromaStore(persist)

    # -- indexing -----------------------------------------------------------
    def index(self, chunks: list[TableChunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = {c.chunk_id: c for c in chunks}
        self.order = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        if self.embedder is not None:
            vecs = self.embedder.encode([chunk_summary(c) for c in chunks], normalize_embeddings=True, batch_size=16)
            self.store.reset(vecs.shape[1])
            self.store.add(self.order, vecs.tolist())

    # -- search -------------------------------------------------------------
    def _semantic_query(self, query: str) -> str:
        q = normalize_periods(query)
        extra = expand_query(q)[len(q):].strip() if self.cfg.get("query_expansion", True) else ""
        return f"{q} (needs: {extra})" if extra else q

    def bm25_ranking(self, query: str) -> list[str]:
        q = normalize_periods(query)
        if self.cfg.get("query_expansion", True):
            q = expand_query(q)
        scores = self.bm25.get_scores(tokenize(q))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.order[i] for i in ranked[: self.cfg["candidates"]] if scores[i] > 0]

    def dense_ranking(self, query: str) -> list[str]:
        if self.embedder is None:
            return []
        vec = self.embedder.encode([BGE_QUERY_PREFIX + self._semantic_query(query)], normalize_embeddings=True)[0]
        return self.store.query(vec.tolist(), self.cfg["candidates"])

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        if not self.chunks:
            return []
        top_k = top_k or self.cfg["top_k"]
        sparse, dense = self.bm25_ranking(query), self.dense_ranking(query)
        k = self.cfg["rrf_k"]
        fused: dict[str, float] = {}
        for ranking in (sparse, dense):
            for r, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + r + 1)
        cands = sorted(fused, key=lambda c: -fused[c])[: self.cfg["rerank_candidates"]]
        if self.reranker is not None and cands:
            q = self._semantic_query(query)
            scores = self.reranker.predict([(q, chunk_summary(self.chunks[c])) for c in cands])
            final = sorted(zip(cands, map(float, scores)), key=lambda x: -x[1])
        else:
            final = [(c, fused[c]) for c in cands]
        return [Hit(self.chunks[c], s, sparse.index(c) + 1 if c in sparse else None,
                    dense.index(c) + 1 if c in dense else None, fused[c]) for c, s in final[:top_k]]


def main() -> None:
    from ingest_financials import ingest_paths

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="+", help="files to index, followed by the question as the last argument")
    ap.add_argument("--bm25-only", action="store_true", help="skip model downloads (sparse search only)")
    args = ap.parse_args()
    *paths, question = args.path
    retriever = HybridRetriever(load_models=not args.bm25_only)
    retriever.index(ingest_paths(paths, retriever.cfg["max_rows_per_chunk"]))
    for i, h in enumerate(retriever.search(question), 1):
        print(f"\n[{i}] score={h.score:.3f} bm25_rank={h.bm25_rank} dense_rank={h.dense_rank}  {h.chunk.chunk_id}")
        print(h.chunk.text)


if __name__ == "__main__":
    main()
