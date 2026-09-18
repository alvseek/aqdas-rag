"""Isolate why the dense backend lost, before accepting that it lost.

The prediction was that dense would beat lexical on the gloss queries by a wide
margin. It lost by 0.155. That is an unpredicted result, which means the first
suspect is the instrument rather than the conclusion -- a retriever can be
genuinely worse, or it can be correctly implemented in the wrong way, and the
two are indistinguishable from the score alone.

Two candidate faults, tested separately:

1. **Query encoding.** BGE retrieval models are trained with an instruction
   prefix on the query side. fastembed exposes `query_embed` for this and
   describes the prefix as "not so necessary" for v1.5 -- a claim about a third
   party's model, which is worth measuring rather than repeating.

2. **Truncation.** The model takes 512 tokens. Any paragraph longer than that
   is silently cut, and the lost tail cannot be retrieved by anything.
"""

from __future__ import annotations

import numpy as np

from aqdas_rag.evalset import build as build_evalset
from aqdas_rag.evaluate import ParagraphCorpus

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOKEN_LIMIT = 512


def normalise(vectors: list) -> np.ndarray:
    matrix = np.array(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def recall_at(matrix: np.ndarray, query_matrix: np.ndarray, golds: list[int], k: int) -> float:
    scores = query_matrix @ matrix.T
    top = np.argsort(-scores, axis=1)[:, :k]
    hits = sum(1 for row, gold in zip(top, golds) if gold in row)
    return hits / len(golds)


def main() -> int:
    from fastembed import TextEmbedding

    corpus = ParagraphCorpus()
    cases = [c for c in build_evalset() if c.difficulty == "gloss"]
    index = {r["citation"]: i for i, r in enumerate(corpus.records)}
    cases = [c for c in cases if c.gold in index]
    golds = [index[c.gold] for c in cases]

    model = TextEmbedding(model_name=MODEL_NAME)

    # --- fault 2 first: it is cheap and changes how fault 1 is read ---------
    tokenizer = getattr(model.model, "tokenizer", None)
    lengths = []
    for record in corpus.records:
        if tokenizer is not None:
            lengths.append(len(tokenizer.encode(record["text"]).ids))
        else:
            lengths.append(int(len(record["text"].split()) * 1.35))

    over = [n for n in lengths if n > TOKEN_LIMIT]
    print(f"paragraphs        : {len(corpus.records)}")
    print(f"longest (tokens)  : {max(lengths)}")
    print(f"over {TOKEN_LIMIT} tokens : {len(over)}  -> silently truncated")

    # --- fault 1: query encoding -------------------------------------------
    doc_texts = [f"{r.get('lemma','')} {r['text']}".strip() for r in corpus.records]
    doc_matrix = normalise(list(model.embed(doc_texts)))

    queries = [c.query for c in cases]
    plain = normalise(list(model.embed(queries)))
    prefixed = normalise(list(model.query_embed(queries)))

    print(f"\ngloss cases       : {len(cases)}")
    for k in (1, 3, 5, 10):
        a = recall_at(doc_matrix, plain, golds, k)
        b = recall_at(doc_matrix, prefixed, golds, k)
        print(f"  R@{k:<3} embed()={a:.3f}   query_embed()={b:.3f}   delta={b - a:+.3f}")

    # --- and the passage side, which has its own prefix --------------------
    passage_matrix = normalise(list(model.passage_embed(doc_texts)))
    print()
    for k in (1, 5):
        c = recall_at(passage_matrix, prefixed, golds, k)
        print(f"  R@{k:<3} passage_embed()+query_embed()={c:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
