"""Retrieval over the Aqdas corpus.

Two things here are deliberate.

**Backends are pluggable.** This is a project for *testing* RAG, so the point
is not to pick a retriever but to be able to compare them. BM25 ships first
because it needs no model download and no API key -- on a 487-record corpus
whose vocabulary is highly distinctive (Ḥuqúqu'lláh, Mashriqu'l-Adhkár, Bayán)
a lexical retriever is a serious baseline, not a toy. A dense backend slots in
beside it behind the same interface.

**Retrieval expands along the book's own structure.** A hit on a paragraph
pulls the Notes that annotate it, because that is how the book is read: the
commentary is not a separate document that happens to be similar, it is
attached. This is the part a generic chunk-and-embed pipeline cannot do, and
it comes from published links rather than from similarity.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"

# BM25 constants. Standard values; k1 controls term-frequency saturation and
# b controls length normalisation.
K1 = 1.5
B = 0.75

# Minimum IDF-mass coverage for a result to be returned at all.
#
# Without a floor, BM25 always returns something: one weak match on a common
# word is enough, so a question about semiconductor foundries comes back with
# ten confident-looking passages and the answering model assumes they are
# relevant. That is the mechanism behind grounded-sounding hallucination, and
# it defeats the promise to say when the book is silent.
#
# The value is measured, not chosen -- see calibrate.py, which scores eighteen
# questions the book answers against six it does not. Measured with stemming on
# and the reader-phrased questions included:
#     worst on-topic  0.335   (what happens to someone who steals)
#     best off-topic  0.247   (best training split for hypertrophy)
# 0.29 sits in that gap.
#
# 🚨 The margin is THIN -- 0.088, down from 0.223 before reader-phrased
# questions were added to the set. That narrowing is the real finding: the
# easy queries borrowed the book's own vocabulary and made the floor look safer
# than it is. A lexical retriever cannot reach a law stated in words the reader
# does not use, and no floor fixes that -- set it high and real questions are
# reported as silent, set it low and off-topic noise returns. This is the
# evidence for adding a dense retriever, not a number to keep tuning.
#
# Re-run the calibration after any change to tokenisation or corpus scope; if
# the two groups ever overlap, no floor is safe and it says so rather than
# suggesting one.
RELEVANCE_FLOOR = 0.29

STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "it", "that", "for", "on", "with",
    "as", "be", "by", "are", "this", "from", "at", "or", "an", "a", "we",
    "what", "does", "do", "say", "says", "about", "how", "who", "which",
}


# Apostrophes must be DELETED, not turned into spaces. Transliterated Bahá'í
# terms carry them inside a single word -- Ḥuqúqu'lláh, Mashriqu'l-Adhkár,
# 'Abdu'l-Bahá -- so replacing them with a space splits one word into two, and
# a reader who types "Huququllah" without the apostrophe then matches neither
# half. Every other punctuation mark is a genuine word boundary.
APOSTROPHES = dict.fromkeys(map(ord, "'‘’ʻʼʿ`"), None)


def fold(text: str) -> str:
    """Normalise for matching: lowercase, strip diacritics, drop punctuation.

    So a question typed as "Huququllah" reaches the passage that spells it
    Ḥuqúqu'lláh. The stored text keeps every diacritic -- only the index is
    folded, because what gets quoted back must stay verbatim.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower().translate(APOSTROPHES))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", stripped)


# Conservative suffix stripping. Without it a reader asking about "drinking"
# reaches nothing, because the translation says "drink" -- the question fails
# on morphology rather than on meaning, and the floor then reports the book as
# silent on something it plainly addresses.
#
# "-eth" is here because the register demands it: this is 19th-century English
# and the text is full of stealeth, behooveth, prescribeth. A general-purpose
# stemmer trained on modern prose does not strip it.
#
# Order matters -- longest suffix first. A strip is rejected if it would leave
# a stub shorter than three characters, which is what stops death -> d and
# teeth -> te.
SUFFIXES = ("eth", "ies", "ing", "ed", "es", "s")
MIN_STEM = 3


def stem(token: str) -> str:
    for suffix in SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM:
            root = token[: -len(suffix)]
            if suffix == "ies":
                return root + "y"
            if suffix == "s" and root.endswith("s"):
                return token  # keep -ss words (witness, holiness) intact
            return root
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in fold(text).split() if t and t not in STOPWORDS]


@dataclass
class Hit:
    record: dict
    score: float
    reason: str  # why this record is in the result set

    @property
    def citation(self) -> str:
        return self.record["citation"]


class Corpus:
    """The parsed volume, plus the structural index that makes it navigable."""

    def __init__(self, path: Path = CORPUS_PATH) -> None:
        self.records: list[dict] = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        self.by_citation = {r["citation"]: r for r in self.records}

        # Paragraph -> the Notes that annotate it, from the published links.
        self.notes_for: dict[str, list[dict]] = defaultdict(list)
        for record in self.records:
            for ref in record.get("annotates", []):
                self.notes_for[ref].append(record)

    def __len__(self) -> int:
        return len(self.records)


class BM25Retriever:
    """Okapi BM25 over the folded text of every record."""

    name = "bm25"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.docs: list[list[str]] = []

        for record in corpus.records:
            # The lemma of a Note is the phrase it annotates, and it is often
            # the most searchable thing about it -- index it alongside the body.
            searchable = f"{record.get('lemma', '')} {record['text']}"
            self.docs.append(tokenize(searchable))

        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = sum(self.doc_len) / max(len(self.docs), 1)

        self.freqs: list[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))

        n_docs = len(self.docs)
        self.idf = {
            term: math.log(1 + (n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def coverage(self, terms: set[str], doc_index: int) -> float:
        """Fraction of the query's rarity-weight present in one document.

        A term the corpus has never seen counts toward the denominator at
        maximum rarity: if the question is about something the book does not
        discuss, that absence is the answer, and a statistic that ignored the
        unknown words would report the query as well covered.

        The live relevance floor and calibrate.py both call this, so the
        measurement that justified the floor and the filter that enforces it
        cannot drift apart.
        """
        if not terms:
            return 0.0

        max_idf = max(self.idf.values()) if self.idf else 1.0
        doc = self.freqs[doc_index]

        total = sum(self.idf.get(t, max_idf) for t in terms)
        matched = sum(self.idf.get(t, max_idf) for t in terms if doc.get(t))
        return matched / total if total else 0.0

    def search(self, query: str, k: int = 5, floor: float = RELEVANCE_FLOOR) -> list[Hit]:
        terms = tokenize(query)
        if not terms:
            return []
        term_set = set(terms)

        scores = [0.0] * len(self.docs)
        for i, freq in enumerate(self.freqs):
            length = self.doc_len[i]
            total = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + K1 * (1 - B + B * length / self.avg_len)
                total += self.idf.get(term, 0.0) * tf * (K1 + 1) / denom
            scores[i] = total

        ranked = sorted(
            (
                (s, i)
                for i, s in enumerate(scores)
                if s > 0 and self.coverage(term_set, i) >= floor
            ),
            reverse=True,
        )[:k]

        return [
            Hit(self.corpus.records[i], round(score, 3), "lexical match")
            for score, i in ranked
        ]


def expand_with_structure(corpus: Corpus, hits: list[Hit]) -> list[Hit]:
    """Attach each retrieved paragraph's Notes, and each Note's paragraph.

    These additions are not ranked results -- they are the surrounding
    apparatus the book itself supplies, and they are labelled as such so the
    answering model can tell commentary from revealed text.
    """
    seen = {h.citation for h in hits}
    expanded = list(hits)

    for hit in hits:
        record = hit.record

        if record["unit_type"] == "paragraph":
            for note in corpus.notes_for.get(record["citation"], []):
                if note["citation"] not in seen:
                    seen.add(note["citation"])
                    expanded.append(
                        Hit(note, 0.0, f"annotates {record['citation']}")
                    )

        elif record["unit_type"] == "note":
            for ref in record.get("annotates", []):
                para = corpus.by_citation.get(ref)
                if para and ref not in seen:
                    seen.add(ref)
                    expanded.append(
                        Hit(para, 0.0, f"annotated by {record['citation']}")
                    )

    return expanded


def search(query: str, k: int = 5, expand: bool = True) -> list[Hit]:
    corpus = Corpus()
    hits = BM25Retriever(corpus).search(query, k=k)
    return expand_with_structure(corpus, hits) if expand else hits


def main() -> int:
    import sys

    if len(sys.argv) < 2:
        print('usage: python -m aqdas_rag.retrieve "your question"')
        return 2

    query = " ".join(sys.argv[1:])
    hits = search(query)

    if not hits:
        print("No passage in the Kitáb-i-Aqdas matched that query.")
        return 0

    print(f'query: {query}\n')
    for hit in hits:
        record = hit.record
        marker = f"[{hit.score}]" if hit.score else f"[+ {hit.reason}]"
        print(f"{record['citation']}  {marker}  — {record['author']}")
        if record.get("lemma"):
            print(f"  on: “{record['lemma'][:90]}”")
        print(f"  {record['text'][:260]}...")
        print(f"  {record['url']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
