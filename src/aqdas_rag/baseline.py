"""Criterion 4: how much of RAG's citation error is retrieval, and how much is generation?

The Aqdas corpus is about 78k tokens, so "paste the entire book" is a real
competitor and the honest ceiling to measure retrieval against. But the two arms
cost very differently -- 77,700 tokens per full-context call against 1,650 for a
retrieved one, a 47x ratio -- and most of what the expensive arm would tell you
can be derived from the cheap one.

RAG citation accuracy is the product of two independent things:

    P(cites gold) = P(gold retrieved) x P(cites gold | gold retrieved)

The first factor is **free** -- a set-membership check on the retrieved records,
no model call. The second needs calls, but only cheap ones. Measure both and you
know which half to invest in; if the conditional rate is near 1.0 then all the
loss is retrieval, full context wins by construction, and the expensive run can
be skipped entirely.

🚨 **The leak this arm had to be fixed for.** Every eval query is derived from a
Note, so that Note self-matches at rank 1, and `expand_with_structure` then
hands over the very paragraph it annotates. Measured before the fix: the source
Note was retrieved 40/40 times and the gold paragraph arrived via expansion in
19 of 40 cases that retrieval had not actually found. The RAG arm was being
handed its answer. `evaluate.py` avoided this with a paragraph-only index and no
expansion; this file reintroduced it by using the full corpus, one module over.

The fix is to exclude the query's originating Note from the retrieved set. No
real reader pastes a Note verbatim and then asks what it refers to, so removing
it restores the condition the measurement is supposed to describe -- while
leaving expansion itself intact, since attaching commentary to a genuinely
retrieved paragraph is the feature, not the leak.

    uv run python -m aqdas_rag.baseline --route claude_cli --sample 40
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from aqdas_rag.evalset import build as build_evalset
from aqdas_rag.hybrid import HybridRetriever
from aqdas_rag.llm import ModelUnavailable, get_model
from aqdas_rag.retrieve import Corpus, expand_with_structure

TOP_K = 5
ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "data" / "baseline-results.json"

SYSTEM = """\
You answer questions about the Kitáb-i-Aqdas using only the passages provided.

Rules:
- Cite every claim with the exact citation given, e.g. ¶63, Note 125, Q&A 8.
- Quote the published wording; do not paraphrase the text into your own words.
- Notes are commentary by the Universal House of Justice, not revealed text.
  Never present one as the other.
- If the passages do not address the question, say so. Do not answer from
  general knowledge.

End your reply with a line of the form:
CITATIONS: ¶12, Note 34
"""

CITATION_RE = re.compile(r"(¶\s*\d+|Note\s+\d+|Q&A\s+\d+)")


@dataclass
class Trial:
    question: str
    gold: str
    arm: str
    gold_retrieved: bool = False
    cited: list[str] = field(default_factory=list)
    correct: bool = False
    error: str = ""


def parse_citations(reply: str) -> list[str]:
    tail = reply.rsplit("CITATIONS:", 1)[-1] if "CITATIONS:" in reply else reply
    seen, out = set(), []
    for raw in CITATION_RE.findall(tail):
        citation = re.sub(r"\s+", " ", raw).strip()
        citation = citation.replace("¶ ", "¶")
        if citation not in seen:
            seen.add(citation)
            out.append(citation)
    return out


def build_rag_context(
    corpus: Corpus, retriever, question: str, exclude: str = ""
) -> tuple[str, list[str]]:
    """Retrieved passages, with the query's own source Note removed.

    Returns the rendered context and the citations it contains, so retrieval
    success is recorded from what the model was actually shown rather than
    re-derived later from a second search that might not match.
    """
    hits = [h for h in retriever.search(question, k=TOP_K) if h.citation != exclude]
    expanded = [h for h in expand_with_structure(corpus, hits) if h.citation != exclude]

    citations = [h.citation for h in expanded]
    rendered = "\n\n".join(
        f"[{h.record['citation']} — {h.record['author']}]\n{h.record['text']}"
        for h in expanded
    )
    return rendered or "(no passage matched)", citations


def full_context(corpus: Corpus) -> str:
    return "\n\n".join(
        f"[{r['citation']} — {r['author']}]\n{r['text']}" for r in corpus.records
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", default="claude_cli",
                        choices=["claude_cli", "openrouter", "anthropic"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", default="rag", choices=["rag", "both"])
    args = parser.parse_args()

    try:
        client = get_model(args.route, args.model)
    except ModelUnavailable as exc:
        print(f"{exc}\n")
        return 2

    corpus = Corpus()
    retriever = HybridRetriever(corpus)

    cases = [c for c in build_evalset() if c.difficulty == "gloss"]
    step = max(1, len(cases) // args.sample)
    cases = cases[::step][: args.sample]

    arms = ("rag",) if args.arms == "rag" else ("rag", "full")
    whole_book = full_context(corpus) if "full" in arms else ""

    # Contexts are built once, before any call, so the retrieval figures are
    # fixed before generation can influence which cases get reported.
    prepared = {}
    for case in cases:
        context, citations = build_rag_context(
            corpus, retriever, case.query, exclude=case.source
        )
        prepared[case.query] = (context, case.gold in citations)

    retrieved = sum(1 for _, ok in prepared.values() if ok)
    print(f"route      : {client.name}")
    print(f"cases      : {len(cases)}   arms: {', '.join(arms)}")
    print(f"gold retrieved (free, no model): {retrieved}/{len(cases)} = "
          f"{retrieved / len(cases):.3f}\n")

    jobs = [(case, arm) for case in cases for arm in arms]
    lock = threading.Lock()
    done = [0]

    def run(job) -> Trial:
        case, arm = job
        context, gold_in = prepared[case.query]
        trial = Trial(case.query, case.gold, arm, gold_retrieved=gold_in)
        if arm == "full":
            context, trial.gold_retrieved = whole_book, True

        try:
            reply = client.complete(
                SYSTEM, f"Passages:\n\n{context}\n\nQuestion: {case.query}"
            )
            trial.cited = parse_citations(reply)
            trial.correct = case.gold in trial.cited
        except Exception as exc:
            trial.error = f"{type(exc).__name__}: {exc}"[:200]

        with lock:
            done[0] += 1
            mark = "ok  " if trial.correct else ("ERR " if trial.error else "miss")
            flag = "" if trial.gold_retrieved else "  (gold not retrieved)"
            print(f"  [{done[0]:>3}/{len(jobs)}] {arm:<4} {case.gold:<6} {mark}{flag}")
        return trial

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        trials = list(pool.map(run, jobs))

    print()
    rag = [t for t in trials if t.arm == "rag" and not t.error]
    with_gold = [t for t in rag if t.gold_retrieved]
    without = [t for t in rag if not t.gold_retrieved]

    def rate(rows: list[Trial]) -> str:
        if not rows:
            return "n/a"
        hit = sum(1 for t in rows if t.correct)
        return f"{hit}/{len(rows)} = {hit / len(rows):.3f}"

    print("  DECOMPOSITION")
    print(f"    retrieval    P(gold retrieved)          : {retrieved}/{len(cases)}"
          f" = {retrieved / len(cases):.3f}")
    print(f"    generation   P(cites | gold retrieved)  : {rate(with_gold)}")
    print(f"    leak check   P(cites | NOT retrieved)   : {rate(without)}"
          "   <- should be ~0")
    print(f"    combined     P(cites gold)              : {rate(rag)}")

    if without:
        leaked = sum(1 for t in without if t.correct)
        if leaked:
            print(f"\n  🚨 {leaked} case(s) cited the gold WITHOUT it being in context --")
            print("     the model is answering from its own knowledge of the text,")
            print("     not from the passages. Citation accuracy overstates grounding.")

    errors = Counter(t.error.split(":")[0] for t in trials if t.error)
    if errors:
        print(f"\n  errors: {dict(errors)}")

    if "full" in arms:
        full = [t for t in trials if t.arm == "full" and not t.error]
        print(f"\n  full-context P(cites gold)              : {rate(full)}")

    RESULTS_PATH.write_text(
        json.dumps([t.__dict__ for t in trials], ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n  detail -> {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
