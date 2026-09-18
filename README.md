# aqdas-rag

A retrieval agent over the **Kitáb-i-Aqdas** that answers with citations you can click
and verify, served to any agent over MCP.

```sh
uv sync
uv run python -m aqdas_rag.fetch        # download the text
uv run python -m aqdas_rag.parse        # build the corpus
uv run python -m aqdas_rag.smoke_test   # 9 checks, expect "all checks passed"
```

Then open Claude Code in this directory — `.mcp.json` loads the server automatically —
and ask. Or query it directly:

```sh
uv run python -m aqdas_rag.retrieve "what does it say about Huququllah"
```

**Full documentation → [docs/README.md](docs/README.md)** — setup, the four MCP tools,
how retrieval works, the design decisions and their measurements, and what is known to be
broken.

The text is © Bahá'í World Centre; it is fetched at build time rather than redistributed.
