"""Offline contracts for the narrow local PixelRAG evaluation runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import eval_runner

_HASH = "a" * 64


def _write_index(root: Path, *, pages: int = 3) -> Path:
    index = root / "synthetic-index"
    tile_dir = index / "tiles" / "0.png.tiles"
    tile_dir.mkdir(parents=True)
    (index / "index.faiss").write_bytes(b"synthetic")
    (index / "articles.json").write_text(json.dumps([{"title": "synthetic", "url": "/synthetic.pdf"}]))
    (index / "summary.json").write_text(json.dumps({"backend": "faiss", "dimension": 8}))
    for page in range(pages):
        (tile_dir / f"tile_{page:04d}.jpg").write_bytes(b"synthetic-image")
    (tile_dir / "tiles.json").write_text(
        json.dumps(
            {
                "source": "/synthetic.pdf",
                "article_id": 0,
                "complete": True,
                "total_pages": pages,
                "tiles": [f"tile_{page:04d}.jpg" for page in range(pages)],
            }
        )
    )
    (tile_dir / "chunks.json").write_text(
        json.dumps(
            {
                "article_id": 0,
                "num_chunks": pages,
                "chunks": [
                    {
                        "tile_index": page,
                        "chunk_index": 0,
                        "file": f"tile_{page:04d}.jpg",
                    }
                    for page in range(pages)
                ],
            }
        )
    )
    np.savez(
        index / "metadata.npz",
        article_ids=np.zeros(pages, dtype=np.int64),
        tile_indices=np.arange(pages, dtype=np.int32),
        chunk_indices=np.zeros(pages, dtype=np.int32),
        y_offsets=np.zeros(pages, dtype=np.int32),
        tile_heights=np.full(pages, 1000, dtype=np.int32),
    )
    (index / "pixelrag-evaluation-index.json").write_text(
        json.dumps(
            {
                "schema_version": eval_runner.INDEX_SCHEMA,
                "index_name": index.name,
                "document_id": "fsae_rules_2024_v1",
                "source_id": "fsae_rules_2024_v1",
                "source_sha256": _HASH,
                "page_count": pages,
                "embedding_model": "local/synthetic-embedder",
                "index_backend": "faiss",
                "created_at": "2026-10-08T00:00:00Z",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return index


class _Searcher:
    def __init__(self, hits: list[dict[str, object]]) -> None:
        self.hits = hits
        self.closed = False

    def search(self, question: str, limit: int) -> list[dict[str, object]]:
        assert question == "SYNTHETIC_QUESTION"
        assert limit == 3
        return self.hits

    def close(self) -> None:
        self.closed = True


def _hits() -> list[dict[str, object]]:
    return [
        {
            "score": 0.99,
            "vector_id": 2,
            "article_id": 0,
            "tile_index": 2,
            "chunk_index": 0,
            "y_offset": 0,
            "tile_height": 1000,
        },
        {
            "score": 0.75,
            "vector_id": 0,
            "article_id": 0,
            "tile_index": 0,
            "chunk_index": 0,
            "y_offset": 0,
            "tile_height": 1000,
        },
        {
            "score": 0.5,
            "vector_id": 1,
            "article_id": 0,
            "tile_index": 1,
            "chunk_index": 0,
            "y_offset": 0,
            "tile_height": 1000,
        },
    ]


def test_query_returns_verified_zero_based_pdf_pages_and_closes_searcher(tmp_path: Path) -> None:
    _write_index(tmp_path)
    searcher = _Searcher(_hits())

    response = eval_runner._query(
        {
            "index_root": str(tmp_path),
            "index_name": "synthetic-index",
            "question": "SYNTHETIC_QUESTION",
            "top_k": 2,
            "mode": "retriever_only",
            "device": "cpu",
        },
        searcher_factory=lambda **_kwargs: searcher,
    )

    assert searcher.closed is True
    assert response["schema_version"] == eval_runner.RUNNER_SCHEMA
    assert response["outcome"] == "retrieval_only"
    assert response["answer"] is None
    assert response["identity"] == {
        "index_name": "synthetic-index",
        "document_id": "fsae_rules_2024_v1",
        "source_id": "fsae_rules_2024_v1",
        "source_sha256": _HASH,
        "page_count": 3,
        "embedding_model": "local/synthetic-embedder",
        "index_backend": "faiss",
        "index_manifest_sha256": hashlib.sha256(
            (tmp_path / "synthetic-index" / "pixelrag-evaluation-index.json").read_bytes()
        ).hexdigest(),
    }
    assert response["results"] == [
        {
            "rank": 1,
            "document_id": "fsae_rules_2024_v1",
            "source_id": "fsae_rules_2024_v1",
            "pdf_page_index": 2,
            "native_page": 3,
            "score": 0.99,
            "article_id": 0,
            "tile_index": 2,
            "chunk_index": 0,
            "y_offset": 0,
            "tile_height": 1000,
            "image": "tiles/0.png.tiles/tile_0002.jpg",
        },
        {
            "rank": 2,
            "document_id": "fsae_rules_2024_v1",
            "source_id": "fsae_rules_2024_v1",
            "pdf_page_index": 0,
            "native_page": 1,
            "score": 0.75,
            "article_id": 0,
            "tile_index": 0,
            "chunk_index": 0,
            "y_offset": 0,
            "tile_height": 1000,
            "image": "tiles/0.png.tiles/tile_0000.jpg",
        },
    ]
    assert response["generation"] == {"requested": False, "model": None, "evidence_ranks": []}


def test_full_rag_uses_only_local_reader_path_and_removes_reasoning(tmp_path: Path) -> None:
    _write_index(tmp_path)
    local_reader = tmp_path / "local-reader"
    local_reader.mkdir()
    received: dict[str, object] = {}

    def generate(**kwargs: object) -> str:
        received.update(kwargs)
        return "<think>private chain of thought</think> SYNTHETIC_ANSWER"

    response = eval_runner._query(
        {
            "index_root": str(tmp_path),
            "index_name": "synthetic-index",
            "question": "SYNTHETIC_QUESTION",
            "top_k": 2,
            "mode": "full_rag",
            "device": "cpu",
            "generation_model": str(local_reader),
            "generation_top_k": 2,
            "max_tokens": 64,
        },
        searcher_factory=lambda **_kwargs: _Searcher(_hits()),
        generator=generate,
    )

    assert response["outcome"] == "answered"
    assert response["answer"] == "SYNTHETIC_ANSWER"
    assert response["generation"] == {
        "requested": True,
        "model": str(local_reader),
        "evidence_ranks": [1, 2],
    }
    assert received["image_paths"] == [
        tmp_path / "synthetic-index" / "tiles/0.png.tiles/tile_0002.jpg",
        tmp_path / "synthetic-index" / "tiles/0.png.tiles/tile_0000.jpg",
    ]


def test_query_rejects_any_non_page_pdf_index_layout(tmp_path: Path) -> None:
    index = _write_index(tmp_path)
    np.savez(
        index / "metadata.npz",
        article_ids=np.zeros(3, dtype=np.int64),
        tile_indices=np.arange(3, dtype=np.int32),
        chunk_indices=np.array([0, 1, 0], dtype=np.int32),
        y_offsets=np.zeros(3, dtype=np.int32),
        tile_heights=np.full(3, 1000, dtype=np.int32),
    )

    with pytest.raises(eval_runner.RunnerInputError, match="index_data_invalid"):
        eval_runner._query(
            {
                "index_root": str(tmp_path),
                "index_name": "synthetic-index",
                "question": "SYNTHETIC_QUESTION",
                "top_k": 1,
                "mode": "retriever_only",
                "device": "cpu",
            },
            searcher_factory=lambda **_kwargs: _Searcher(_hits()),
        )


def test_atomic_index_build_validation_uses_the_target_name_not_the_temporary_directory(tmp_path: Path) -> None:
    index = _write_index(tmp_path)
    build_path = tmp_path / ".synthetic-index.build-random"
    index.rename(build_path)

    manifest = eval_runner._load_index(build_path, expected_index_name="synthetic-index")

    assert manifest.index_name == "synthetic-index"
    with pytest.raises(eval_runner.RunnerInputError, match="index_manifest_invalid"):
        eval_runner._load_index(build_path)


def test_full_rag_rejects_a_nonlocal_reader(tmp_path: Path) -> None:
    _write_index(tmp_path)

    with pytest.raises(eval_runner.RunnerInputError, match="generation_model_not_local"):
        eval_runner._query(
            {
                "index_root": str(tmp_path),
                "index_name": "synthetic-index",
                "question": "SYNTHETIC_QUESTION",
                "top_k": 1,
                "mode": "full_rag",
                "device": "cpu",
                "generation_model": "Qwen/Qwen3-VL-4B-Instruct",
                "generation_top_k": 1,
                "max_tokens": 64,
            },
            searcher_factory=lambda **_kwargs: _Searcher(_hits()),
        )
