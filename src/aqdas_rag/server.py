"""MCP server exposing the Kitáb-i-Aqdas as a citable source.

The retrieval quality matters less than the honesty of the surface. Any agent
can be told "cite your sources" in a prompt and will drift from it within a few
turns; what does not drift is a tool that *returns* citations as structured
data and cannot return a passage without one.

So the contract here is: every unit that comes back carries its own citation,
its author, and a URL that resolves to the official published text. Nothing is
summarised on the way out -- the caller receives the words as published and
does its own reasoning, which is the only arrangement in which "quote verbatim"
is checkable rather than merely promised.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from aqdas_rag.hybrid import HybridRetriever
from aqdas_rag.retrieve import Corpus, expand_with_structure

INSTRUCTIONS = """\
This server serves the Kitáb-i-Aqdas (the Most Holy Book of Bahá'u'lláh),
authorized English translation, from the Bahá'í Reference Library.

How to use it well:

- **Cite every claim.** Each result carries a `citation` (¶N, Note N, Q&A N)
  and a `url`. Put the citation next to the statement it supports.
- **Quote, do not paraphrase.** The `text` field is the published wording.
  When stating what the book says, quote it and mark your own words as yours.
- **Do not blur authorship.** The `author` field matters: ¶ and Q&A are
  Bahá'u'lláh's revealed text; Notes are commentary by the Universal House of
  Justice; the Synopsis is Shoghi Effendi's codification. Commentary is not
  revelation, and presenting one as the other is a citation error even when
  the words are accurate.
- **Say when the book is silent.** If a search returns nothing relevant, say
  the Kitáb-i-Aqdas does not address the question rather than reaching for a
  general answer. This book is one text among many in the Bahá'í writings.
- **This server reports, it does not rule.** The Aqdas is a book of law with
  an authoritative interpretive tradition. Present what the text says and what
  the Notes say about it; leave rulings to the institutions that make them.
"""

mcp = FastMCP(name="aqdas", instructions=INSTRUCTIONS)

_corpus = Corpus()
_retriever = HybridRetriever(_corpus)


def _render(record: dict, reason: str = "", score: float = 0.0) -> dict[str, Any]:
    out = {
        "citation": record["citation"],
        "author": record["author"],
        "text": record["text"],
        "url": record["url"],
        "unit_type": record["unit_type"],
    }
    if record.get("lemma"):
        out["annotates_phrase"] = record["lemma"]
    if record.get("annotates"):
        out["annotates"] = record["annotates"]
    if reason:
        out["included_because"] = reason
    if score:
        out["score"] = score
    return out


@mcp.tool(
    annotations={"readOnlyHint": True, "openWorldHint": False},
)
def search(query: str, limit: int = 20) -> dict[str, Any]:
    """Search the Kitáb-i-Aqdas for passages relevant to a question.

    Returns the matching paragraphs, Notes and Questions-and-Answers, each with
    its citation, author and a URL to the official text. Results are expanded
    along the book's own structure: a matching paragraph brings the Notes that
    annotate it, and a matching Note brings the paragraph it annotates.

    An empty `results` list means the search found nothing -- report that the
    book does not appear to address the question rather than answering from
    general knowledge.

    `limit` defaults to 20 because retrieval depth was measured to be the
    dominant lever on this corpus: recall of the correct paragraph runs 0.600
    at limit=5 and 0.825 at limit=20, for 6.5k tokens of context instead of
    1.4k. Aqdas paragraphs are short, so depth is nearly free here -- raising
    it bought three times what improving the ranker did.
    """
    hits = _retriever.search(query, k=limit)
    if not hits:
        return {
            "query": query,
            "results": [],
            "note": "No passage matched. Say the Kitáb-i-Aqdas does not appear "
                    "to address this rather than answering from general knowledge.",
        }

    expanded = expand_with_structure(_corpus, hits)
    return {
        "query": query,
        "results": [
            _render(h.record, "" if h.score else h.reason, h.score) for h in expanded
        ],
    }


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
def read(citation: str) -> dict[str, Any]:
    """Read one unit verbatim by its citation.

    Accepts the forms used in Bahá'í scholarship: "¶63", "63", "Note 125",
    "Q&A 8". Use this when a citation is already known and the exact published
    wording is needed.
    """
    key = citation.strip()
    candidates = [key]
    if key.isdigit():
        candidates.append(f"¶{key}")
    if not key.startswith("¶"):
        candidates.extend([f"Note {key}", f"Q&A {key}"])

    for candidate in candidates:
        record = _corpus.by_citation.get(candidate)
        if record:
            result = _render(record)
            if record["unit_type"] == "paragraph":
                result["notes_on_this"] = [
                    _render(n) for n in _corpus.notes_for.get(record["citation"], [])
                ]
            return result

    return {
        "error": f"No unit is cited as {citation!r}.",
        "valid_forms": ["¶1 .. ¶190", "Note 1 .. Note 194", "Q&A 1 .. Q&A 103"],
    }


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
def notes_on(paragraph: str) -> dict[str, Any]:
    """List the Notes annotating a given paragraph, e.g. "¶63" or "63".

    The Notes are the Universal House of Justice's commentary on specific
    phrases; they are what turns a terse legal verse into something a reader
    can follow, and they are keyed to the paragraph by the publisher.
    """
    key = paragraph.strip()
    key = key if key.startswith("¶") else f"¶{key.lstrip('¶')}"

    if key not in _corpus.by_citation:
        return {"error": f"No paragraph is cited as {paragraph!r}.",
                "valid_range": "¶1 .. ¶190"}

    notes = _corpus.notes_for.get(key, [])
    return {
        "paragraph": key,
        "count": len(notes),
        "notes": [_render(n) for n in notes],
        "note": "" if notes else f"{key} carries no annotation in the published Notes.",
    }


@mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
def corpus_stats() -> dict[str, Any]:
    """Report what this server actually holds -- counted from the corpus."""
    from collections import Counter

    by_type = Counter(r["unit_type"] for r in _corpus.records)
    annotated = len(_corpus.notes_for)
    return {
        "source": "Bahá'í Reference Library, authorized English translation",
        "units": dict(sorted(by_type.items())),
        "total_units": len(_corpus),
        "paragraphs_with_notes": annotated,
        "contains": "Kitáb-i-Aqdas main text, the Notes, and Questions and Answers",
        "does_not_contain": [
            "the Introduction and Preface",
            "the Synopsis and Codification",
            "Supplementary Texts",
            "the wider Bahá'í writings",
        ],
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
