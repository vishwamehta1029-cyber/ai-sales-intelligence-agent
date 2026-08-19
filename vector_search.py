"""
Lightweight vector search index over the knowledge_base/ docs.

Design note (kept honest in the README too): this uses TF-IDF vectors +
cosine similarity rather than a hosted embedding model, so it runs fully
offline with no API key. Documents are still represented as vectors in a
similarity index and retrieved by nearest-neighbor search — the same
retrieval pattern used by a hosted vector DB (Pinecone, pgvector,
Snowflake Cortex Search, etc). Swapping in real embeddings later is a
localized change: replace `TfidfVectorizer` with an embeddings client
and `cosine_similarity` with the vector DB's query call.
"""

import glob
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorSearchIndex:
    def __init__(self, docs_dir="knowledge_base"):
        self.docs_dir = docs_dir
        self.chunks = []       # list of (source_file, text_chunk)
        self.vectorizer = None
        self.matrix = None
        self._build()

    def _chunk(self, text, max_chars=280):
        # naive paragraph/bullet-level chunking, good enough for this demo
        parts = [p.strip() for p in text.split("\n") if p.strip() and not p.strip().startswith("#")]
        chunks, buf = [], ""
        for p in parts:
            if len(buf) + len(p) > max_chars and buf:
                chunks.append(buf.strip())
                buf = ""
            buf += " " + p
        if buf.strip():
            chunks.append(buf.strip())
        return chunks

    def _build(self):
        for path in sorted(glob.glob(os.path.join(self.docs_dir, "*.md"))):
            with open(path) as f:
                text = f.read()
            for chunk in self._chunk(text):
                self.chunks.append((os.path.basename(path), chunk))

        corpus = [c for _, c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, k=2):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [
            {"source": self.chunks[i][0], "text": self.chunks[i][1], "score": round(float(sims[i]), 3)}
            for i in top_idx if sims[i] > 0
        ]


if __name__ == "__main__":
    idx = VectorSearchIndex()
    for q in ["what discount needs VP approval", "how are territories defined"]:
        print(f"\nQuery: {q}")
        for r in idx.search(q):
            print(f"  [{r['source']} score={r['score']}] {r['text'][:120]}...")
