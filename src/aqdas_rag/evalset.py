"""Derive a retrieval test set from the book's own published cross-references.

The Synopsis and Codification looked like the obvious answer key -- it is a
topical classification of every law -- but the Reference Library edition
carries no paragraph references in it, only the outline. So it cannot serve.

The Notes can, and better. Each Note is published with two facts: the exact
phrase it annotates (its lemma) and a link to the paragraph that phrase sits
in. That yields verified (question, expected paragraph) pairs that nobody on
this project wrote, which is the whole point -- an eval set I author tests my
own assumptions about what the retriever should find.

Two difficulties are generated from the same links:

**lemma** -- the query is the annotated phrase itself. The words are literally
present in the target paragraph, so any lexical retriever should score near
perfect. A miss here is a bug, not a limitation.

**gloss** -- the query is the Note's own opening sentence, which explains the
phrase in the House of Justice's modern prose rather than repeating
Bahá'u'lláh's wording. The target paragraph usually shares few content words
with it. This is the vocabulary gap that decides whether retrieval actually
works for a reader who does not already know the text's phrasing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
OUT_PATH = ROOT / "data" / "evalset.jsonl"

MIN_QUERY_WORDS = 4


@dataclass(frozen=True)
class EvalCase:
    difficulty: str          # "lemma" | "gloss"
    query: str
    gold: str                # the paragraph citation that must be retrieved
    source: str              # the Note the pair was derived from


def first_sentence(text: str) -> str:
    """Opening sentence of a Note body, trimmed to something query-shaped."""
    # Abbreviations that must not be read as a sentence end.
    guarded = re.sub(r"\b(Mr|Mrs|Dr|St|cf|e\.g|i\.e)\.", r"\1<DOT>", text)
    match = re.search(r"^(.{40,320}?[.!?])\s", guarded + " ")
    sentence = match.group(1) if match else guarded[:240]
    return sentence.replace("<DOT>", ".").strip()


def build() -> list[EvalCase]:
    records = [
        json.loads(line)
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]
    paragraphs = {r["citation"] for r in records if r["unit_type"] == "paragraph"}
    cases: list[EvalCase] = []

    for record in records:
        if record["unit_type"] != "note":
            continue

        # Only Notes pointing at exactly one paragraph give an unambiguous
        # gold answer. A Note annotating three paragraphs has no single
        # correct retrieval, and scoring it either way would be arbitrary.
        targets = [t for t in record.get("annotates", []) if t in paragraphs]
        if len(targets) != 1:
            continue
        gold = targets[0]

        lemma = record.get("lemma", "").strip()
        if len(lemma.split()) >= MIN_QUERY_WORDS:
            cases.append(EvalCase("lemma", lemma, gold, record["citation"]))

        gloss = first_sentence(record["text"])
        if len(gloss.split()) >= MIN_QUERY_WORDS:
            cases.append(EvalCase("gloss", gloss, gold, record["citation"]))

    return cases


def load() -> list[EvalCase]:
    return [
        EvalCase(**json.loads(line))
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    cases = build()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for case in cases:
            fh.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")

    from collections import Counter

    by_difficulty = Counter(c.difficulty for c in cases)
    golds = {c.gold for c in cases}

    print(f"eval cases written : {len(cases):,} -> {OUT_PATH}")
    for difficulty, count in sorted(by_difficulty.items()):
        print(f"  {difficulty:<8} {count:>5}")
    print(f"distinct gold paragraphs : {len(golds)}")

    for difficulty in ("lemma", "gloss"):
        sample = next((c for c in cases if c.difficulty == difficulty), None)
        if sample:
            print(f"\n  [{difficulty}] gold={sample.gold}  from {sample.source}")
            print(f"    {sample.query[:150]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
