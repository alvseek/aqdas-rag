"""Measure what separates an on-topic query from an off-topic one.

A relevance floor is needed, but picking the number by feel is how a pipeline
acquires a constant nobody can defend later. So this measures the separation on
real queries first, and the floor is read off the gap rather than typed into it.

The statistic is **IDF-mass coverage**: of the total rarity-weight of the terms
in the question, what fraction is actually present in the retrieved unit. Raw
BM25 scores cannot serve here because they drift with corpus size and document
length; a coverage fraction is bounded in [0, 1] and means the same thing
whatever the corpus does next.

Terms absent from the corpus entirely count toward the denominator at maximum
rarity -- if someone asks about semiconductor foundries, the book's failure to
contain that word is the whole point, and a statistic that ignored it would
score the query as well-covered.
"""

from __future__ import annotations

from aqdas_rag.retrieve import BM25Retriever, Corpus, tokenize

# Questions the Kitáb-i-Aqdas genuinely addresses.
#
# The first group borrows the book's own vocabulary, which made the original
# calibration too easy -- every one scored near 1.0 and the floor looked safer
# than it was. The second group is how a reader actually asks: ordinary modern
# words for laws the text states in 19th-century English. Those are the cases
# that decide whether a floor protects against hallucination or manufactures
# false silence, so they belong in the measurement.
ON_TOPIC = [
    "what does it say about Huququllah",
    "marriage and consent of parents",
    "the law of inheritance when someone dies without a will",
    "obligatory prayer and the Qiblih",
    "fasting and who is exempt",
    "the Mashriqul-Adhkar house of worship",
    "punishment for theft",
    "the Universal House of Justice",
    "burial and how far the body may be carried",
    "pilgrimage",
    "use of opium and intoxicating drink",
    "the Nineteen Day Feast",
    # reader-phrased
    "is drinking wine forbidden",
    "can I gamble",
    "how many wives may a man have",
    "what happens to someone who steals",
    "do I have to pray every day",
    "is music allowed",
]

# Questions it does not address at all.
OFF_TOPIC = [
    "quarterly earnings guidance semiconductor foundry",
    "how do I configure a kubernetes ingress controller",
    "best training split for hypertrophy",
    "what is the airspeed velocity of an unladen swallow",
    "typescript generic type inference rules",
    "how to reheat pizza without it going soggy",
]


def best_coverage(retriever: BM25Retriever, query: str) -> tuple[float, str]:
    """Coverage of the best-covered document for this query.

    Uses the retriever's own coverage method -- the same one the live floor
    enforces -- so this calibration measures the thing it is calibrating
    rather than a reimplementation that happens to agree today.
    """
    terms = set(tokenize(query))
    best, best_cite = 0.0, "-"
    for i, record in enumerate(retriever.corpus.records):
        c = retriever.coverage(terms, i)
        if c > best:
            best, best_cite = c, record["citation"]
    return best, best_cite


def main() -> int:
    retriever = BM25Retriever(Corpus())

    print("ON-TOPIC (should score high)")
    on_scores = []
    for q in ON_TOPIC:
        c, cite = best_coverage(retriever, q)
        on_scores.append(c)
        print(f"  {c:5.2f}  {cite:<10} {q}")

    print("\nOFF-TOPIC (should score low)")
    off_scores = []
    for q in OFF_TOPIC:
        c, cite = best_coverage(retriever, q)
        off_scores.append(c)
        print(f"  {c:5.2f}  {cite:<10} {q}")

    lo_on, hi_off = min(on_scores), max(off_scores)
    print(f"\nworst on-topic : {lo_on:.3f}")
    print(f"best off-topic : {hi_off:.3f}")

    if lo_on > hi_off:
        floor = round((lo_on + hi_off) / 2, 2)
        print(f"separation     : CLEAN (gap {lo_on - hi_off:.3f})")
        print(f"suggested floor: {floor}")
    else:
        print("separation     : OVERLAPPING -- no single floor separates these")
        print("                 a floor here would silence real questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
