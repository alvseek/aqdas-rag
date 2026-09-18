---
doc_type: 7q-readme
---

# aqdas-rag

## Table of Contents

- [What Is This?](#what-is-this)
- [How Do I Set It Up?](#how-do-i-set-it-up)
- [How Do I Use It?](#how-do-i-use-it)
- [How Does It Work Inside?](#how-does-it-work-inside)
- [How Is It Deployed?](#how-is-it-deployed)
- [What Decisions Were Made?](#what-decisions-were-made)
- [What's Broken / Known Debts?](#whats-broken--known-debts)

---

## What Is This?

A retrieval agent over the **Kitáb-i-Aqdas** that answers questions with citations you
can click and verify against the official published text. It is served over MCP, so any
agent — including Claude Code itself — can query the book as a tool.

It exists because this book already has a citation system, and the job is mostly
refusing to destroy it. Bahá'í scholarship cites the Aqdas by numbered paragraph (¶63),
by Note, and by Question and Answer, and the Bahá'í Reference Library publishes those
numbers as markup rather than as rendered decoration. So a chunk here is never a window
of N tokens — it is the unit a reader would actually cite.

### Architecture

```
  bahai.org Reference Library
        │  fetch.py      17 HTML fragments, UTF-8 verified
        ▼
  data/raw/*.html
        │  parse.py      shape-aware: paragraph / note / qa handlers
        ▼
  data/corpus.jsonl      487 citable records, each with author + URL
        │
        ├──► retrieve.py   BM25 + stemming + relevance floor
        ├──► dense.py      bge-base-en-v1.5, ONNX/CPU, cached vectors
        │         │
        │         └──► hybrid.py    reciprocal rank fusion  ◄── production
        │                    │
        │                    ▼
        └──────────────► server.py   MCP: search · read · notes_on · corpus_stats

  measurement lane (not in the serving path)
     evalset.py ──► evaluate.py ──► significance.py
     calibrate.py · diagnose_dense.py · baseline.py · llm.py
```

Two lanes matter here. The **serving lane** (`fetch → parse → hybrid → server`) is what
answers questions. The **measurement lane** exists because this is a project for
*testing* RAG — every configuration claim in this document is backed by a script that
can be re-run to falsify it.

### Tech Stack

- **Runtime**: Python ≥ 3.11, managed with `uv`
- **MCP**: FastMCP ≥ 4.0.5
- **Parsing**: BeautifulSoup 4 + lxml
- **Embeddings**: fastembed (ONNX Runtime, CPU) + numpy — no PyTorch
- **Store**: JSONL + a pickled numpy matrix. No database.

---

## How Do I Set It Up?

### Prerequisites

- Python 3.11+ and `uv` (`uv --version`)
- Network access to `bahai.org` and HuggingFace on first run
- No GPU, no API key, no database

### Setup

1. Install dependencies:
   ```sh
   cd C:\Work\research\aqdas-rag
   uv sync
   ```

2. Fetch the text (17 fragments, ~1 MB, polite 1 s delay between requests):
   ```sh
   uv run python -m aqdas_rag.fetch
   ```

3. Build the corpus:
   ```sh
   uv run python -m aqdas_rag.parse
   ```

4. Verify it works:
   ```sh
   uv run python -m aqdas_rag.smoke_test
   # Expected: 9 checks, ending "all checks passed"
   ```

The first dense query downloads the embedding model (~0.21 GB) and builds the vector
index, which takes several minutes on CPU. It is cached in `data/embeddings/` and only
rebuilds when the model or the corpus text changes.

### Environment Variables

None are required. All three are optional:

| Variable | Description | Example |
|----------|-------------|---------|
| `AQDAS_EMBED_MODEL` | Swap the dense backend | `BAAI/bge-large-en-v1.5` |
| `ANTHROPIC_API_KEY` | Only for `baseline.py --route anthropic` | `sk-ant-...` |
| `OPENROUTER_API_KEY` | Only for `baseline.py --route openrouter` | `sk-or-...` |

Put keys in a `.env` file at the project root — it is gitignored, and `llm.py` reads it
with no dependency. Do not use `setx`, which writes the credential permanently into the
Windows user profile where every later process inherits it.

---

## How Do I Use It?

### As an MCP server (the main way)

`.mcp.json` is already configured. Open Claude Code in this directory and the server
loads automatically; to use it from anywhere, copy that server block into your global
MCP config. Then ask questions in plain language — the agent calls the tools itself.

| Tool | What it does |
|------|--------------|
| `search(query, limit=20)` | Find passages relevant to a question, expanded along the book's own Note links |
| `read(citation)` | One unit verbatim — accepts `¶63`, `63`, `Note 125`, `Q&A 8` |
| `notes_on(paragraph)` | The Universal House of Justice's commentary on a given paragraph |
| `corpus_stats()` | What the server actually holds, counted from the corpus |

**What to expect.** Roughly 1 question in 6 phrased in everyday words will miss. When it
misses, it usually does *not* report silence — it answers from adjacent passages. The
citations are always real and clickable, so treat a confident answer as *"here is a
passage near your question"* rather than *"here is the passage"*, and click through.

It has never been observed to invent a citation: in every measured case where the
correct paragraph was absent from context, it declined to cite it.

### From the command line

| Command | Description |
|---------|-------------|
| `uv run python -m aqdas_rag.retrieve "<question>"` | Search and print cited passages |
| `uv run python -m aqdas_rag.smoke_test` | 9 checks over a real MCP session |
| `uv run python -m aqdas_rag.evaluate` | Compare bm25 / dense / hybrid on 369 cases |
| `uv run python -m aqdas_rag.significance` | McNemar test on the paired outcomes |
| `uv run python -m aqdas_rag.calibrate` | Re-derive the relevance floor |
| `uv run python -m aqdas_rag.llm` | Report which model routes are usable |
| `uv run python -m aqdas_rag.baseline --sample 40` | RAG accuracy, split into retrieval vs generation |

---

## How Does It Work Inside?

### Core flow: a question becomes cited passages

1. **Fuse two rankings** (`hybrid.py`)
   - BM25 over folded, stemmed text and dense cosine over `bge-base` vectors are each
     taken to depth 50, then combined by reciprocal rank fusion. Ranks are fused rather
     than scores, because BM25 is unbounded and cosine is bounded — normalising them
     against each other would need a conversion nobody can justify.

2. **Gate on topicality** (`retrieve.py`)
   - A measured floor on IDF-mass coverage decides whether the book discusses the subject
     at all. It is applied on the **lexical** side only: an embedding model returns a
     confident nearest neighbour for any input, including a question about semiconductors,
     so a dense-side floor would need a threshold on a score with no natural zero.
     Nothing clearing the gate means an empty result and an instruction to say so.

3. **Expand along the book's own links** (`retrieve.py`)
   - A retrieved paragraph pulls the Notes that annotate it; a retrieved Note pulls the
     paragraph it annotates. These come from published cross-references, not similarity,
     and are labelled `included_because` so the answering agent can tell them from ranked
     results.

4. **Render with provenance** (`server.py`)
   - Every unit returns its citation, its author, and a URL. Nothing is summarised on the
     way out — the caller gets the published words and does its own reasoning, which is
     the only arrangement in which "quote verbatim" is checkable rather than promised.

### Data model

One flat record type. The structure lives in the fields, not in a hierarchy:

```
Record
  unit_type   paragraph | note | qa
  unit_id     "63"
  citation    "¶63"                    ← the retrieval key AND the human citation
  author      Bahá'u'lláh | Universal House of Justice
  text        verbatim published wording
  url         https://www.bahai.org/...#523504832
  lemma       (notes) the phrase annotated · (qa) the question
  annotates   (notes) ["¶63"]          ← the graph edge
```

Counted from `data/corpus.jsonl`:

| Unit | Count | Author |
|---|---|---|
| Paragraphs ¶1–¶190 | 190 | Bahá'u'lláh |
| Notes | 194 | Universal House of Justice |
| Questions and Answers | 103 | Bahá'u'lláh |

487 units, ~49k words. All 194 Note→¶ links resolve; 114 of 190 paragraphs carry at
least one Note.

### External integrations

| Service | Purpose | Protocol | Timeout |
|---------|---------|----------|---------|
| bahai.org Reference Library | Source text, fetched at build time only | HTTPS | 30 s |
| HuggingFace Hub | Embedding model download, first run only | HTTPS | default |
| Anthropic / OpenRouter | `baseline.py` only, never the serving path | HTTPS / subprocess | 120–180 s |

The serving path makes **no network calls**. Everything it needs is on disk.

---

## How Is It Deployed?

Not deployed. It runs locally as a stdio MCP server launched by the MCP client:

```json
{"command": "uv", "args": ["run", "--directory", "C:\\Work\\research\\aqdas-rag",
                            "python", "-m", "aqdas_rag.server"]}
```

Intended target is a CPU-only cloud server, which the design already accommodates: query
embedding costs 7.8 ms and the serving path needs no GPU and no network. The one
deployment cost to plan for is the **first-run index build** — several minutes on CPU —
so ship `data/embeddings/` with the image or build it once at deploy rather than on
first request.

[TODO: target host, process manager, and whether the MCP transport will stay stdio or move to HTTP]

---

## What Decisions Were Made?

### ADR-001: The chunk is the citation unit, not a token window (2026-09-18)

**Context**: Generic RAG chunks by token count, which makes the retrieved span
uncitable — a citation has to be reconstructed from offsets, and "¶63" degrades into
"somewhere around here".
**Decision**: One record per published unit — paragraph, Note, Q&A — read from the
Reference Library's own `brl-pnum` markup.
**Trade-off**: Records vary in length (9 to 512 tokens), which is untidy for embedding
and means one Note exceeds the model's limit. Accepted, because retrieval cannot lose a
citation it did not have to invent.

### ADR-002: Note→paragraph edges are read, never inferred (2026-09-18)

**Context**: Commentary is only useful when attached to what it explains.
**Decision**: Parse the publisher's own links (`<a href=".../5#304611242">¶4</a>`) rather
than inferring the mapping from proximity or similarity.
**Trade-off**: Depends on upstream markup staying stable. Accepted — a parse failure is
loud, whereas an inferred mapping is a guess wearing a number.

### ADR-003: The relevance floor is measured, and gates lexically (2026-09-18)

**Context**: A retriever always returns something, which is the mechanism behind
grounded-sounding hallucination.
**Decision**: Derive the floor from a calibration run (18 on-topic vs 6 off-topic
questions) rather than choosing a number, and apply it on the BM25 side even under
hybrid retrieval.
**Trade-off**: The separation is thin — 0.335 worst on-topic against 0.247 best
off-topic. Accepted, and `calibrate.py` reports overlap rather than suggesting a number
when no floor is safe.

### ADR-004: Hybrid fusion in production, on a significance test (2026-09-18)

**Context**: BM25 alone missed ¶63 entirely for *"how many wives may a man have"*.
**Decision**: Fuse BM25 and dense by RRF. Hybrid beats BM25 at rank 1 — 20 cases to 8,
exact McNemar **p = 0.036** — and is even at R@5 (11 to 11, p = 1.0).
**Trade-off**: Adds a model download and a multi-minute index build to deployment, for a
win confined to rank 1. Accepted because an agent reads the top hit.

### ADR-005: Retrieval depth over ranker quality (2026-09-18)

**Context**: `limit=5` was an unexamined default.
**Decision**: Default to 20. Recall of the correct paragraph runs 0.600 at 5 and 0.825 at
20, for 6.5k tokens of context instead of 1.4k.
**Trade-off**: Larger tool results. Accepted — Aqdas paragraphs are short, so depth is
nearly free, and raising it bought roughly three times what improving the ranker did.

### ADR-006: The server reports, it does not rule (2026-09-18)

**Context**: The Aqdas is a book of law with an authoritative interpretive tradition.
**Decision**: Tools return published wording with citations and authorship; the server
instructions tell the calling agent to quote rather than paraphrase, never to present
commentary as revelation, and to leave rulings to the institutions that make them.
**Trade-off**: Answers are less fluent than a summary would be. Accepted deliberately.

---

## What's Broken / Known Debts?

**Dangerous — it answers confidently when it should stay silent.** Measured over 40
reader-phrased questions: when the correct paragraph was not retrieved (16 cases), it
admitted silence in only **4** and answered from adjacent passages in **12**. The
relevance floor cannot help, because the retriever came back full rather than empty.
Whether those 12 answers are actually *wrong* has not been checked — they cite real
passages accurately — but the system did not signal uncertainty.

**Retrieval is the bottleneck, and it is where any further work belongs.**
`P(cites gold) = P(retrieved) × P(cites | retrieved) = 0.600 × 0.833 = 0.500` at
`limit=5`. Generation was never the problem.

**The cost/accuracy curve past limit=20 is projected, not measured.** The projection
holds generation fixed at 0.833, which is known to be false — generation degrades with
context length, which is exactly why whole-book context scores 0.625 despite perfect
retrieval. One RAG run at `limit=20` (~260k tokens) would replace the projection with a
measurement.

**The whole-book baseline rests on 8 cases.** It anchors every RAG-versus-full-context
comparison in this document and is the weakest number here. A 40-case run costs ~3.1M
input tokens.

**Seven Notes are truncated in the dense index.** Notes 38, 160, 170, 23, 56, 183 and 86
exceed the model's 512-token limit, the longest at 656 words. This got worse on purpose:
repairing the blockquote parsing added ~19k characters of quoted commentary, and the
Notes that gained most are the ones that now overflow. BM25 indexes them whole, so fusion
partly covers the loss, but a passage quoted deep inside a long Note is reachable only
lexically.

**Three sections of the volume are fetched but not indexed** — the Introduction and
Preface, the Synopsis and Codification, and the Supplementary Texts. So a question the
Introduction answers gets "the Aqdas does not address this", which is true of the main
text and misleading about the book.

**The Synopsis cannot serve as ground truth.** It looked like a free authoritative answer
key — a topical classification of every law — but the Reference Library edition carries
the outline with no paragraph references. The eval set is derived from Note→¶ links
instead, which is better anyway: the gloss half tests the vocabulary gap the Synopsis
could not have.

**Nothing is committed to git.**

---

## On the subject matter

The text is © Bahá'í World Centre. It is fetched at build time rather than redistributed,
and `data/` is not committed.
