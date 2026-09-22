"""Smoke test: exercise the MCP server over a real in-memory client session.

Calling the Python functions directly would prove only that the functions work.
This goes through FastMCP's client so the tools are invoked the way an agent
invokes them -- registration, schema, serialisation and all.

Each check states what it expects BEFORE it runs, so a pass means the predicted
thing happened rather than merely that nothing raised. A check that cannot fail
is not a check.
"""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client

from aqdas_rag.server import mcp


def unwrap(result) -> dict:
    """Pull the structured payload out of a tool result."""
    if getattr(result, "structured_content", None):
        return result.structured_content
    for block in result.content:
        if getattr(block, "text", None):
            return json.loads(block.text)
    raise AssertionError("tool returned no readable content")


async def run() -> int:
    failures: list[str] = []

    def check(label: str, expectation: str, passed: bool, detail: str = "") -> None:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label}")
        print(f"         expected: {expectation}")
        if detail:
            print(f"         actual  : {detail}")
        if not passed:
            failures.append(label)

    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = sorted(t.name for t in tools)
        check(
            "tools are registered",
            "search, read, notes_on, corpus_stats",
            names == ["corpus_stats", "notes_on", "read", "search"],
            ", ".join(names),
        )

        # --- a question the book definitely answers -------------------------
        res = unwrap(await client.call_tool("search", {"query": "marriage and consent of parents"}))
        cites = [r["citation"] for r in res["results"]]
        check(
            "search finds marriage law",
            "at least one ¶ result, each carrying a bahai.org url",
            bool(res["results"])
            and all(r["url"].startswith("https://www.bahai.org/") for r in res["results"]),
            f"{len(cites)} results: {cites[:6]}",
        )

        # --- structural expansion must actually fire ------------------------
        expanded = [r for r in res["results"] if r.get("included_because")]
        check(
            "structural expansion attaches commentary",
            "at least one result included because it annotates another",
            bool(expanded),
            f"{len(expanded)} expanded: {[r['citation'] for r in expanded][:5]}",
        )

        # --- authorship must be distinguished -------------------------------
        authors = {r["author"] for r in res["results"]}
        check(
            "authorship is distinguished",
            "revealed text and commentary carry different authors",
            len(authors) > 1,
            str(sorted(authors)),
        )

        # --- verbatim read by citation --------------------------------------
        para = unwrap(await client.call_tool("read", {"citation": "¶1"}))
        check(
            "read ¶1 returns the opening verse verbatim",
            "text begins 'The first duty prescribed by God'",
            para.get("text", "").startswith("The first duty prescribed by God"),
            para.get("text", "")[:60],
        )

        # --- a citation that does not exist must fail loudly ----------------
        bad = unwrap(await client.call_tool("read", {"citation": "¶999"}))
        check(
            "an invalid citation is refused",
            "an error, not an invented passage",
            "error" in bad,
            str(bad)[:80],
        )

        # --- pure RAG always returns something, silence is the reader's call --
        empty = unwrap(
            await client.call_tool(
                "search", {"query": "quarterly earnings guidance semiconductor foundry"}
            )
        )
        check(
            "an off-topic query returns passages (pure RAG, no gate)",
            "non-empty results; the reader model decides silence, not retrieval",
            len(empty["results"]) > 0 and "note" not in empty,
            f"{len(empty['results'])} results",
        )

        # --- notes_on, on a paragraph known to carry several ----------------
        notes = unwrap(await client.call_tool("notes_on", {"paragraph": "16"}))
        check(
            "notes_on ¶16 returns its commentary",
            "count > 1, every note authored by the Universal House of Justice",
            notes.get("count", 0) > 1
            and all(n["author"] == "Universal House of Justice" for n in notes["notes"]),
            f"count={notes.get('count')}",
        )

        stats = unwrap(await client.call_tool("corpus_stats", {}))
        check(
            "corpus_stats reports derived counts",
            "190 paragraphs, 194 notes, 103 q&a",
            stats["units"] == {"note": 194, "paragraph": 190, "qa": 103},
            str(stats["units"]),
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
