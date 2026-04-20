"""Paper search index: embed chunks and retrieve by cosine similarity.

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings. The index
is built once during spec creation, cached to disk as a pickle, and
loaded by each phase sub-agent's search_paper tool.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from ..llm.paper import extract_full_text
from .chunker import Chunk, chunk_paper

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Cache the model in-process so repeated search() calls don't reload it.
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(name: str) -> SentenceTransformer:
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


@dataclass
class SearchResult:
    """A single search hit."""

    chunk: Chunk
    score: float

    def format(self) -> str:
        pages = (
            f"page {self.chunk.page_start}"
            if self.chunk.page_start == self.chunk.page_end
            else f"pages {self.chunk.page_start}-{self.chunk.page_end}"
        )
        return (
            f"--- (score: {self.score:.2f}, {pages}, "
            f'section: "{self.chunk.section_heading}") ---\n'
            f"{self.chunk.text}"
        )


@dataclass
class PaperIndex:
    """Semantic search index over a research paper.

    Attributes:
        chunks: The text chunks with metadata.
        embeddings: Tensor of shape (n_chunks, embed_dim).
        model_name: Name of the sentence-transformers model used.
        paper_hash: SHA-256 of the source PDF (for cache invalidation).
    """

    chunks: list[Chunk]
    embeddings: torch.Tensor
    model_name: str = _DEFAULT_MODEL
    paper_hash: str = ""

    @classmethod
    def build(cls, paper_path: Path, model_name: str = _DEFAULT_MODEL) -> PaperIndex:
        """Build an index from a PDF paper.

        Args:
            paper_path: Path to the research paper PDF.
            model_name: Sentence-transformers model to use.

        Returns:
            A PaperIndex ready for search.
        """
        logger.info("Building paper index from %s", paper_path)

        paper_hash = _hash_file(paper_path)
        full_text = extract_full_text(paper_path)
        chunks = chunk_paper(full_text)

        if not chunks:
            logger.warning("No chunks produced from paper")
            return cls(chunks=[], embeddings=torch.empty(0), paper_hash=paper_hash)

        logger.info("Chunked paper into %d passages, embedding with %s", len(chunks), model_name)
        model = _get_model(model_name)
        texts = [c.text for c in chunks]
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        logger.info("Index built: %d chunks, embedding dim %d", len(chunks), embeddings.shape[1])
        return cls(
            chunks=chunks,
            embeddings=embeddings,
            model_name=model_name,
            paper_hash=paper_hash,
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Find the most relevant chunks for a query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of SearchResult sorted by descending relevance.
        """
        if not self.chunks:
            return []

        model = _get_model(self.model_name)
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Ensure both tensors are on the same device.
        embeddings = self.embeddings.to(query_embedding.device)

        # Cosine similarity.
        scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), embeddings, dim=1
        )

        top_k = min(top_k, len(self.chunks))
        top_indices = torch.topk(scores, k=top_k).indices.tolist()

        return [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def save(self, path: Path) -> None:
        """Persist the index to disk."""
        # Move embeddings to CPU for pickling.
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings.cpu(),
            "model_name": self.model_name,
            "paper_hash": self.paper_hash,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Index saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> PaperIndex:
        """Load an index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            chunks=data["chunks"],
            embeddings=data["embeddings"],
            model_name=data["model_name"],
            paper_hash=data.get("paper_hash", ""),
        )

    @classmethod
    def load_or_build(
        cls,
        paper_path: Path,
        index_path: Path,
        model_name: str = _DEFAULT_MODEL,
    ) -> PaperIndex:
        """Load a cached index if it matches the paper, otherwise rebuild.

        Args:
            paper_path: Path to the PDF.
            index_path: Where to cache the index pickle.
            model_name: Sentence-transformers model name.

        Returns:
            A PaperIndex, either loaded from cache or freshly built.
        """
        paper_hash = _hash_file(paper_path)

        if index_path.exists():
            try:
                index = cls.load(index_path)
                if index.paper_hash == paper_hash:
                    logger.info("Loaded cached paper index from %s", index_path)
                    return index
                logger.info("Paper changed (hash mismatch), rebuilding index")
            except Exception:
                logger.warning("Failed to load cached index, rebuilding")

        index = cls.build(paper_path, model_name)
        index.save(index_path)
        return index


def _hash_file(path: Path) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()
