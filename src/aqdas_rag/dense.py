"""Dense retrieval backend, behind the same interface as BM25.

The point of adding this is not that embeddings are better -- it is that the
lexical retriever has a measured failure the eval set makes visible, and a
second backend is the only way to find out whether that failure is a property
of lexical matching or of the corpus.

Uses fastembed: ONNX runtime rather than PyTorch, so the model is a ~130MB
download instead of a multi-gigabyte one, and it runs on CPU at a speed that
makes a 487-document corpus a non-event.

Embeddings are cached to disk keyed by **model name plus a hash of the corpus
text**. A cache keyed only on the filename would silently serve vectors for the
previous parse after any change upstream -- the retriever would work, the
numbers would be real, and they would describe a corpus that no longer exists.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

from aqdas_rag.retrieve import Corpus, Hit

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "embeddings"

# Swappable. The cache key hashes the model name with the corpus text, so
# changing this re-embeds rather than silently serving the old model's vectors.
#   bge-small-en-v1.5  384 dim  0.07 GB  -- first tested, lost to BM25
#   bge-base-en-v1.5   768 dim  0.21 GB  -- current
MODEL_NAME = os.environ.get("AQDAS_EMBED_MODEL", "BAAI/bge-base-en-v1.5")


def _corpus_fingerprint(corpus: Corpus, model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for record in corpus.records:
        digest.update(record["citation"].encode("utf-8"))
        digest.update(record["text"].encode("utf-8"))
    return digest.hexdigest()[:16]


class DenseRetriever:
    """Cosine similarity over sentence embeddings."""

    name = "dense"

    def __init__(self, corpus: Corpus, model_name: str = MODEL_NAME) -> None:
        self.corpus = corpus
        self.model_name = model_name
        self._model = None  # loaded lazily; queries need it, a cache hit may not

        fingerprint = _corpus_fingerprint(corpus, model_name)
        cache_path = CACHE_DIR / f"{fingerprint}.pkl"

        if cache_path.exists():
            self.matrix = pickle.loads(cache_path.read_bytes())
        else:
            texts = [
                # The lemma of a Note names the phrase it explains and is often
                # the most retrievable thing about it -- embed it with the body,
                # matching what the BM25 index is built over so the two backends
                # are compared on the same content.
                f"{r.get('lemma', '')} {r['text']}".strip()
                for r in corpus.records
            ]
            vectors = np.array(list(self._embed(texts)), dtype=np.float32)
            self.matrix = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(pickle.dumps(self.matrix))

    def _load_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _embed(self, texts: list[str]):
        return self._load_model().embed(texts)

    def search(self, query: str, k: int = 5, floor: float = 0.0) -> list[Hit]:
        vector = np.array(next(iter(self._embed([query]))), dtype=np.float32)
        vector /= np.linalg.norm(vector)

        scores = self.matrix @ vector
        order = np.argsort(-scores)[:k]

        return [
            Hit(self.corpus.records[i], round(float(scores[i]), 3), "semantic match")
            for i in order
            if scores[i] >= floor
        ]


def main() -> int:
    import sys

    corpus = Corpus()
    print(f"embedding {len(corpus)} records with {MODEL_NAME} ...")
    retriever = DenseRetriever(corpus)
    print(f"matrix: {retriever.matrix.shape}")

    query = " ".join(sys.argv[1:]) or "what happens to someone who steals"
    print(f"\nquery: {query}\n")
    for hit in retriever.search(query, k=5):
        print(f"  {hit.score:6.3f}  {hit.citation:<10} {hit.record['text'][:90]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
