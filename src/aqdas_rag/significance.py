"""Is the hybrid retriever's win over BM25 real, or is it 12 lucky cases?

With bge-base the fused ranker leads BM25 by 0.061 on gloss R@1. On 194 cases
that is about twelve questions, and a difference that small is exactly the kind
a mechanism story will happily explain before anyone checks whether it survives
being counted properly.

The right test for paired binary outcomes is McNemar's, because the two rankers
are scored on the *same* questions. What matters is not how many each got right
but the **discordant** pairs -- the cases where exactly one of them succeeded.
The cases both got right, and both got wrong, carry no information about which
is better, and including them is what makes a difference look more solid than
it is.

Exact binomial rather than the chi-square approximation: the discordant counts
here are small enough that the approximation is not trustworthy, and an exact
test costs nothing at this size.
"""

from __future__ import annotations

from math import comb

from aqdas_rag.dense import DenseRetriever, MODEL_NAME
from aqdas_rag.evalset import build as build_evalset
from aqdas_rag.evaluate import ParagraphCorpus, rank_of_gold
from aqdas_rag.hybrid import HybridRetriever
from aqdas_rag.retrieve import BM25Retriever

DEPTH = 10


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(name_a: str, ranks_a: list[int | None],
            name_b: str, ranks_b: list[int | None], k: int) -> None:
    hit_a = [r is not None and r <= k for r in ranks_a]
    hit_b = [r is not None and r <= k for r in ranks_b]

    only_a = sum(1 for a, b in zip(hit_a, hit_b) if a and not b)
    only_b = sum(1 for a, b in zip(hit_a, hit_b) if b and not a)
    both = sum(1 for a, b in zip(hit_a, hit_b) if a and b)
    neither = sum(1 for a, b in zip(hit_a, hit_b) if not a and not b)

    p = mcnemar_exact(only_a, only_b)
    verdict = "significant" if p < 0.05 else "NOT significant"

    print(f"\n  {name_a} vs {name_b}  @R{k}")
    print(f"    both right {both:>4}   both wrong {neither:>4}   (carry no information)")
    print(f"    only {name_a:<7} {only_a:>4}   only {name_b:<7} {only_b:>4}   (the evidence)")
    print(f"    exact McNemar p = {p:.4f}  -> {verdict}")


def main() -> int:
    cases = [c for c in build_evalset() if c.difficulty == "gloss"]
    corpus = ParagraphCorpus()
    cases = [c for c in cases if c.gold in corpus.by_citation]

    print(f"embedding model : {MODEL_NAME}")
    print(f"gloss cases     : {len(cases)}")

    backends = {
        "bm25": BM25Retriever(corpus),
        "dense": DenseRetriever(corpus),
        "hybrid": HybridRetriever(corpus),
    }

    ranks: dict[str, list[int | None]] = {}
    for name, retriever in backends.items():
        ranks[name] = [rank_of_gold(retriever, c, DEPTH) for c in cases]

    for k in (1, 5):
        compare("hybrid", ranks["hybrid"], "bm25", ranks["bm25"], k)
    compare("bm25", ranks["bm25"], "dense", ranks["dense"], 5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
