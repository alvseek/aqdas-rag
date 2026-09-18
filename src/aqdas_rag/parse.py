"""Parse the fetched Reference Library fragments into citable records.

The whole project rests on one property of this book: it is already divided
into units that Bahá'í scholarship cites by number, and the Reference Library
publishes those numbers as markup rather than as rendered decoration. So a
"chunk" here is never a window of N tokens -- it is a paragraph, a Note, or a
Question and Answer, exactly as a reader would cite it.

Three things this module refuses to guess:

1. **Which fragment holds what.** Derived from each page's own heading, not
   from a hardcoded map, so an upstream renumbering surfaces as a changed
   report instead of a silent mis-parse.

2. **Which paragraph a Note annotates.** The Notes carry real links back into
   the main text (``<a href=".../5#304611242">¶4</a>``). We read those links.
   Inferring the mapping from proximity would be a guess wearing a number.

3. **Who wrote a unit.** The volume interleaves revealed text, the Universal
   House of Justice's Notes, and Shoghi Effendi's codification. An agent that
   cites the third as though it were the first is not citing well, however
   correct its paragraph number is.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

BASE_URL = "https://www.bahai.org/library/authoritative-texts/bahaullah/kitab-i-aqdas"

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "corpus.jsonl"

# Who authored each kind of unit. This is a property of the published volume,
# not a heuristic -- the 1992 edition states it in its own front matter.
AUTHOR_BY_TYPE = {
    "paragraph": "Bahá'u'lláh",
    "qa": "Bahá'u'lláh",
    "note": "Universal House of Justice",
    "synopsis": "Shoghi Effendi",
    "frontmatter": "Universal House of Justice",
}

# Section headings we recognise, mapped to the shape-handler that reads them.
# A heading we do not recognise is reported, never silently dropped.
SECTION_SHAPES = {
    "the kitáb-i-aqdas": "numbered",
    "questions and answers": "qa",
    "notes": "note",
}

PARA_REF_RE = re.compile(r"¶\s*(\d+)")
NOTE_HEAD_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$", re.S)


@dataclass
class Record:
    """One citable unit of the published volume."""

    unit_type: str
    unit_id: str
    citation: str
    author: str
    text: str
    url: str
    fragment: int
    anchor: str = ""
    lemma: str = ""
    annotates: list[str] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


# Tags that sit INSIDE a word and must not introduce a space when flattened.
# The transliteration in this text underlines digraphs -- mi<u>th</u>qáls is
# one word, and <u>th</u> is a single letter of the Bahá'í orthography, not
# emphasis. Extracting with a separator at every tag boundary turns that into
# "mi th qáls", which is no longer the verbatim text the agent promises to
# quote. Block-level boundaries still get their space.
INLINE_TAGS = ("u", "span", "em", "i", "b", "strong", "small", "sub", "sup", "a")


def clean_text(node: Tag) -> str:
    """Visible text of a node, with navigation anchors removed, word-accurate."""
    clone = BeautifulSoup(str(node), "lxml")
    for junk in clone.select("a.brl-pnum, a.brl-location, sup.brl-note"):
        junk.decompose()
    for inline in clone.find_all(list(INLINE_TAGS)):
        inline.unwrap()
    # unwrap() removes the tag but leaves its text as a SEPARATE string node,
    # and get_text(separator) still separates those -- so the space comes back
    # unless adjacent strings are merged into one first.
    clone.smooth()
    text = clone.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def section_heading(soup: BeautifulSoup) -> str:
    head = soup.select_one("h2.brl-head.brl-title, h1.brl-doc-title")
    return head.get_text(" ", strip=True).lower() if head else ""


def nearest_anchor(node: Tag) -> str:
    loc = node.select_one("a.brl-location")
    return loc.get("id", "") if loc else ""


def parse_numbered(soup: BeautifulSoup, fragment: int) -> list[Record]:
    """Main text: a record starts at each numeric paragraph number.

    Accumulates until the next numbered unit, so a paragraph split across
    several elements (verse line-groups, block quotes) stays one record.
    """
    records: list[Record] = []
    current: Record | None = None

    for node in soup.select("div.brl-annotated, div[data-fragment]"):
        for el in node.find_all(["p", "h2", "h3", "div"], recursive=True):
            pnum = el.find("a", class_="brl-pnum", recursive=False)
            number = pnum.get_text(strip=True) if pnum else ""

            if number.isdigit():
                if current is not None:
                    records.append(current)
                anchor = nearest_anchor(el)
                current = Record(
                    unit_type="paragraph",
                    unit_id=number,
                    citation=f"¶{number}",
                    author=AUTHOR_BY_TYPE["paragraph"],
                    text=clean_text(el),
                    url=f"{BASE_URL}/{fragment}#{anchor}",
                    fragment=fragment,
                    anchor=anchor,
                )
            elif current is not None and el.name == "p":
                extra = clean_text(el)
                if extra:
                    current.text = f"{current.text} {extra}".strip()
        break  # the first matching container holds the fragment's body

    if current is not None:
        records.append(current)
    return records


def parse_notes(soup: BeautifulSoup, fragment: int) -> list[Record]:
    """Notes: each ``div[data-unit=section]`` is one Note.

    The Note's own header carries its number, the lemma it annotates, and one
    or more links back into the main text. Those links are the edge that lets
    a retrieved paragraph pull its commentary.
    """
    records: list[Record] = []

    for section in soup.select("div[data-unit='section']"):
        header = section.select_one("p.brl-text-larger1")
        if header is None:
            continue

        bold = header.select_one("span.brl-bold")
        if bold is None:
            continue

        match = NOTE_HEAD_RE.match(bold.get_text(" ", strip=True))
        if not match:
            continue

        number, lemma = match.group(1), match.group(2).strip()

        # Paragraph back-references, read from the published links.
        annotates: list[str] = []
        for link in header.select("a[href]"):
            for ref in PARA_REF_RE.findall(link.get_text(" ", strip=True)):
                if f"¶{ref}" not in annotates:
                    annotates.append(f"¶{ref}")

        # Every <p> in the section, at any depth, in document order. The
        # Reference Library wraps quoted passages -- Shoghi Effendi,
        # 'Abdu'l-Bahá, the House of Justice -- in a nested block, so a
        # direct-children-only walk silently drops the quotation while
        # leaving a note that still reads as complete. Measured: 24 Notes
        # lost ~19k characters that way, including the clarification that
        # polygamy is not permitted (Note 89).
        body_parts = [
            clean_text(p)
            for p in section.find_all("p", recursive=True)
            if p is not header and header not in p.parents
        ]
        body = " ".join(part for part in body_parts if part).strip()

        anchor = nearest_anchor(header)
        records.append(
            Record(
                unit_type="note",
                unit_id=number,
                citation=f"Note {number}",
                author=AUTHOR_BY_TYPE["note"],
                text=body,
                url=f"{BASE_URL}/{fragment}#{anchor}",
                fragment=fragment,
                anchor=anchor,
                lemma=lemma,
                annotates=annotates,
            )
        )

    return records


def parse_qa(soup: BeautifulSoup, fragment: int) -> list[Record]:
    """Questions and Answers: number, question, answer -- kept as one record.

    Splitting the question from its answer would make either half uncitable;
    the pair is the unit a reader refers to.
    """
    records: list[Record] = []

    for section in soup.select("div[data-unit='section']"):
        num_el = section.select_one("p.brl-margin-number")
        if num_el is None:
            continue

        number = num_el.get_text(" ", strip=True).rstrip(". ").strip()
        if not number.isdigit():
            continue

        question, answer, anchor = "", "", ""
        for p in section.find_all("p", recursive=True):
            if p is num_el or num_el in p.parents:
                continue
            text = clean_text(p)
            if text.startswith("Question:"):
                question = text[len("Question:"):].strip()
                anchor = anchor or nearest_anchor(p)
            elif text.startswith("Answer:"):
                answer = text[len("Answer:"):].strip()
            elif answer:
                answer = f"{answer} {text}".strip()

        if not (question or answer):
            continue

        records.append(
            Record(
                unit_type="qa",
                unit_id=number,
                citation=f"Q&A {number}",
                author=AUTHOR_BY_TYPE["qa"],
                text=f"Question: {question}\nAnswer: {answer}".strip(),
                url=f"{BASE_URL}/{fragment}#{anchor}",
                fragment=fragment,
                anchor=anchor,
                lemma=question,
            )
        )

    return records


SHAPE_HANDLERS = {
    "numbered": parse_numbered,
    "note": parse_notes,
    "qa": parse_qa,
}


def parse_all() -> tuple[list[Record], list[str]]:
    records: list[Record] = []
    unhandled: list[str] = []

    for path in sorted(RAW_DIR.glob("fragment-*.html")):
        fragment = int(path.stem.split("-")[1])
        soup = BeautifulSoup(path.read_bytes().decode("utf-8"), "lxml")
        heading = section_heading(soup)

        shape = SECTION_SHAPES.get(heading)
        if shape is None:
            # Continuation fragments carry no heading of their own; they
            # inherit the shape of the section they continue.
            shape = _inherited_shape(records)
            if not heading and shape:
                pass
            else:
                unhandled.append(f"fragment {fragment}: '{heading or '(no heading)'}'")
                continue

        found = SHAPE_HANDLERS[shape](soup, fragment)
        records.extend(found)

    return records, unhandled


def _inherited_shape(records: list[Record]) -> str | None:
    if not records:
        return None
    last = records[-1].unit_type
    return {"paragraph": "numbered", "note": "note", "qa": "qa"}.get(last)


def main() -> int:
    records, unhandled = parse_all()

    by_type = Counter(r.unit_type for r in records)
    linked_notes = [r for r in records if r.unit_type == "note" and r.annotates]
    paragraphs = sorted(
        (int(r.unit_id) for r in records if r.unit_type == "paragraph")
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Every figure below is counted from the records themselves.
    print(f"records written : {len(records):,}  -> {OUT_PATH}")
    for unit_type, count in sorted(by_type.items()):
        print(f"  {unit_type:<12} {count:>5}")

    if paragraphs:
        expected = set(range(paragraphs[0], paragraphs[-1] + 1))
        missing = sorted(expected - set(paragraphs))
        print(f"\nparagraph range : {paragraphs[0]}..{paragraphs[-1]}")
        print(f"gaps in range   : {missing if missing else 'none'}")

    notes_total = by_type.get("note", 0)
    print(f"notes with a ¶ link : {len(linked_notes)} of {notes_total}")

    words = sum(r.n_words for r in records)
    print(f"total words     : {words:,}  (~{int(words * 1.35):,} tokens)")

    if unhandled:
        print("\nunhandled fragments:")
        for item in unhandled:
            print(f"  {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
