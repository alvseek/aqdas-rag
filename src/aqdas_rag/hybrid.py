"""Hybrid retrieval: reciprocal rank fusion of the lexical and dense backends.

Worth testing because the two fail differently. BM25 misses when the reader's
words differ from the book's; embeddings miss when the distinguishing signal is
a rare proper noun that a small general-purpose model has weak representations
for. Where two retrievers fail on different queries, fusing their rankings can
beat both -- and where one is simply better everywhere, fusion will show that
too by failing to improve on it.

Reciprocal rank fusion is used rather than score blending because the two
backends produce incomparable scales: BM25 is unbounded and corpus-dependent,
cosine similarity is bounded in [-1, 1]. Normalising them against each other
would require a conversion nobody can justify, whereas ranks are already
comparable and RRF needs no tuning beyond its damping constant.
"""

from __future__ import annotations

from aqdas_rag.dense import DenseRetriever
from aqdas_rag.retrieve import RELEVANCE_FLOOR, BM25Retriever, Corpus, Hit

# Standard RRF damping. Large enough that the top few ranks do not dominate
# outright, which is the behaviour that makes fusion robust to one backend
# being confidently wrong.
RRF_K = 60

# How deep each backend is asked to go before fusing. A document that neither
# backend ranks in its top 50 is not going to be rescued by fusion.
FUSION_DEPTH = 50


class HybridRetriever:
    """Rank-fuses BM25 and dense results."""

    name = "hybrid"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.lexical = BM25Retriever(corpus)
        self.semantic = DenseRetriever(corpus)

    def search(self, query: str, k: int = 5, floor: float = RELEVANCE_FLOOR) -> list[Hit]:
        # Pure-RAG mode 2026-09-22: no topicality gate. The floor is off
        # (RELEVANCE_FLOOR = 0.0), and a lexical miss no longer blocks the
        # dense backend -- every backend's ranking fuses, and whatever scores
        # highest returns. Silence is the reader model's decision, not ours.
        lexical_hits = self.lexical.search(query, k=FUSION_DEPTH, floor=floor)
        rankings = [
            lexical_hits,
            self.semantic.search(query, k=FUSION_DEPTH, floor=0.0),
        ]
        rankings = [r for r in rankings if r]
        if not rankings:
            return []

        fused: dict[str, float] = {}
        record_by_citation: dict[str, dict] = {}

        for hits in rankings:
            for position, hit in enumerate(hits, start=1):
                citation = hit.citation
                fused[citation] = fused.get(citation, 0.0) + 1.0 / (RRF_K + position)
                record_by_citation.setdefault(citation, hit.record)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Hit(record_by_citation[citation], round(score, 5), "lexical + semantic")
            for citation, score in ordered
        ]
