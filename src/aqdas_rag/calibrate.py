"""Measure what separates an on-topic query from an off-topic one.

Whether a relevance floor can work at all is a question to measure, not assume.
This reports the separation on a real query set, so the answer is read off data
rather than chosen by feel.

The off-topic questions deliberately come in two kinds, because the difference
between them is the finding. The easy group is from another universe of
discourse and separates cleanly, which makes a floor look safe. The hard group
asks about things the book does not answer using the book's own register --
believer, prayer, company, tax -- and scores high enough to invert the
separation. With both groups in place this reports OVERLAPPING and suggests no
floor, which is the honest answer and the reason the gate is off (ADR-007).
Measured 2026-09-24: BM25 best off-topic 0.619 against a 0.335 worst on-topic;
dense cosine 0.713 against 0.648, the -0.065 gap recorded in ADR-007.

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

# Questions it does not address, in two groups that behave very differently.
#
# The first group is from another universe of discourse. Its overlap with the
# corpus is near zero, so it separates cleanly and makes a floor look safe.
#
# The second group is the one that matters. These are also questions the Aqdas
# does not answer, but they are asked in the book's own register -- ordinary
# words it knows (believer, prayer, company, tax) -- so IDF-mass coverage scores
# them high. On this group the separation goes negative and the floor fails.
# Keep both: the gap between the two groups IS the finding, because a retriever
# measures resemblance to the corpus, not topicality.
OFF_TOPIC_EASY = [
    "quarterly earnings guidance semiconductor foundry",
    "how do I configure a kubernetes ingress controller",
    "best training split for hypertrophy",
    "what is the airspeed velocity of an unladen swallow",
    "typescript generic type inference rules",
    "how to reheat pizza without it going soggy",
]

# The hard negatives -- see the note above and ADR-007.
OFF_TOPIC_HARD = [
    "is a believer permitted to serve in the army",
    "what insurance should a small business carry",
    "how long does copyright last on a book",
    "should a believer join a political party",
    "how do I set up a limited company",
    "how much income tax do I owe",
    "who should I vote for in the election",
    "how do I apply for a university scholarship",
]

# The union, for anything that just wants "questions the book does not answer".
OFF_TOPIC = OFF_TOPIC_EASY + OFF_TOPIC_HARD


def best_coverage(retriever: BM25Retriever, query: str) -> tuple[float, str]:
    """Coverage of the best-covered document for this query.

    Uses the retriever's own coverage method -- the same one the retriever's
    `floor` parameter consults -- so this measures the thing it is calibrating
    rather than a reimplementation that happens to agree today.
    """
    terms = set(tokenize(query))
    best, best_cite = 0.0, "-"
    for i, record in enumerate(retriever.corpus.records):
        c = retriever.coverage(terms, i)
        if c > best:
            best, best_cite = c, record["citation"]
    return best, best_cite


def score_group(retriever: BM25Retriever, label: str,
                queries: list[str]) -> list[float]:
    print(f"\n{label}")
    scores = []
    for q in queries:
        c, cite = best_coverage(retriever, q)
        scores.append(c)
        print(f"  {c:5.2f}  {cite:<10} {q}")
    return scores


def main() -> int:
    retriever = BM25Retriever(Corpus())

    on_scores = score_group(retriever, "ON-TOPIC (should score high)", ON_TOPIC)
    easy_off = score_group(
        retriever, "OFF-TOPIC, another universe of discourse (easy)", OFF_TOPIC_EASY
    )
    hard_off = score_group(
        retriever, "OFF-TOPIC, in the book's own register (hard)", OFF_TOPIC_HARD
    )

    lo_on = min(on_scores)
    print(f"\nworst on-topic : {lo_on:.3f}")
    print(f"best easy off  : {max(easy_off):.3f}   separation {lo_on - max(easy_off):+.3f}")
    print(f"best hard off  : {max(hard_off):.3f}   separation {lo_on - max(hard_off):+.3f}")

    hi_off = max(max(easy_off), max(hard_off))
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
