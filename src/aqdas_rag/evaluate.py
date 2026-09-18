"""Compare retrieval backends on the book's own cross-references.

Two design choices keep this honest.

**The index is paragraphs only.** Every eval query is derived from a Note, so
searching the full corpus would retrieve that Note as a trivial self-match, and
the structural expansion would then hand over the gold paragraph attached to
it. The retriever would score near perfect while demonstrating nothing. Asking
"find the paragraph this question is about" over a paragraph-only index has no
such leak.

**Predictions are stated before the run.** A benchmark that merely prints
numbers invites a mechanism story for whatever it produced; naming the expected
result first means a surprise is visible as a surprise. The predictions are
recorded in PREDICTIONS below and echoed in the report.

The production server searches everything and expands along the Note links,
which can only add to what is measured here -- so these figures are a floor on
the deployed behaviour, not a description of it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from aqdas_rag.dense import DenseRetriever
from aqdas_rag.evalset import EvalCase, build as build_evalset
from aqdas_rag.hybrid import HybridRetriever
from aqdas_rag.retrieve import BM25Retriever, Corpus

KS = (1, 3, 5, 10)

PREDICTIONS = """\
Stated before running:
  lemma  -- the query's words are literally in the target paragraph, so BM25
            should be strong (recall@5 above ~0.85). A miss here is a bug.
  gloss  -- the query explains the phrase in different words, so BM25 should
            fall sharply (recall@5 below ~0.50) and dense should beat it by a
            wide margin. If dense does NOT win here, the embedding step is
            adding cost for nothing and should be dropped.
"""


class ParagraphCorpus(Corpus):
    """The corpus restricted to the main text, for leak-free evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.records = [r for r in self.records if r["unit_type"] == "paragraph"]
        self.by_citation = {r["citation"]: r for r in self.records}


@dataclass
class Score:
    n: int = 0
    hits_at: dict[int, int] = None
    reciprocal_rank: float = 0.0

    def __post_init__(self) -> None:
        if self.hits_at is None:
            self.hits_at = {k: 0 for k in KS}

    def record(self, rank: int | None) -> None:
        self.n += 1
        if rank is None:
            return
        self.reciprocal_rank += 1.0 / rank
        for k in KS:
            if rank <= k:
                self.hits_at[k] += 1

    def recall(self, k: int) -> float:
        return self.hits_at[k] / self.n if self.n else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank / self.n if self.n else 0.0


def rank_of_gold(retriever, case: EvalCase, depth: int) -> int | None:
    # floor=0: the relevance floor is a separate safety mechanism, and applying
    # it here would conflate "could not find it" with "declined to answer".
    hits = retriever.search(case.query, k=depth, floor=0.0)
    for position, hit in enumerate(hits, start=1):
        if hit.citation == case.gold:
            return position
    return None


def evaluate(retriever, cases: list[EvalCase]) -> dict[str, Score]:
    scores: dict[str, Score] = defaultdict(Score)
    depth = max(KS)
    for case in cases:
        rank = rank_of_gold(retriever, case, depth)
        scores[case.difficulty].record(rank)
        scores["overall"].record(rank)
    return scores


def main() -> int:
    cases = build_evalset()
    corpus = ParagraphCorpus()

    print(PREDICTIONS)
    print(f"index      : {len(corpus)} paragraphs (main text only, no leak)")
    print(f"eval cases : {len(cases)}")

    backends = [
        ("bm25", BM25Retriever(corpus)),
        ("dense", DenseRetriever(corpus)),
        ("hybrid", HybridRetriever(corpus)),
    ]

    results: dict[str, dict[str, Score]] = {}
    for name, retriever in backends:
        print(f"\nrunning {name} ...")
        results[name] = evaluate(retriever, cases)

    header = f"\n{'difficulty':<10} {'backend':<7} " + " ".join(
        f"R@{k:<4}" for k in KS
    ) + "  MRR"
    print(header)
    print("-" * len(header))

    for difficulty in ("lemma", "gloss", "overall"):
        for name, _ in backends:
            score = results[name].get(difficulty)
            if not score or not score.n:
                continue
            cells = " ".join(f"{score.recall(k):<6.3f}" for k in KS)
            print(f"{difficulty:<10} {name:<7} {cells}  {score.mrr:.3f}")
        print()

    # The comparison that decides whether the dense backend earns its place.
    bm25_gloss = results["bm25"]["gloss"].recall(5)
    dense_gloss = results["dense"]["gloss"].recall(5)
    delta = dense_gloss - bm25_gloss
    print(f"gloss R@5: bm25 {bm25_gloss:.3f} -> dense {dense_gloss:.3f}  ({delta:+.3f})")

    if delta > 0.10:
        print("verdict: dense earns its place on the vocabulary-gap queries, as predicted.")
    elif delta > 0:
        print("verdict: dense wins but only narrowly -- weaker than predicted; "
              "check whether the cost is worth it.")
    else:
        print("verdict: UNPREDICTED -- dense does not beat lexical here. "
              "Treat the embedding step as unjustified until this is explained.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
