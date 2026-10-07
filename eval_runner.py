#!/usr/bin/env python3
"""Explicit local PixelRAG index/query runner for research evaluation.

The runner is intentionally a short-lived CLI boundary rather than a search
service.  Index creation is an explicit command; query requests only inspect a
published local index and never create, repair, replace, or activate one.

The supported evaluation index shape is deliberately narrow: one complete PDF
per index, rendered as one visual tile per physical PDF page.  That lets the
runner return verified, zero-based physical-page evidence instead of guessing
from generic PixelRAG tile coordinates.  All PDF pages, rendered tiles, FAISS
files, model caches, and output artifacts remain outside Git.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np

RUNNER_SCHEMA = "pixelrag-local-eval-runner-v1"
INDEX_SCHEMA = "pixelrag-local-eval-index-v1"
_INDEX_MANIFEST = "pixelrag-evaluation-index.json"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_ABSTENTION = "INSUFFICIENT_EVIDENCE"


class RunnerInputError(ValueError):
    """A rejected request or index with a stable, secret-free code."""


class Searcher(Protocol):
    def search(self, question: str, limit: int) -> list[Mapping[str, object]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class IndexManifest:
    index_name: str
    document_id: str
    source_id: str
    source_sha256: str
    page_count: int
    embedding_model: str
    index_backend: str
    manifest_sha256: str


@dataclass(frozen=True)
class GenerationConfig:
    model: Path
    image_count: int
    max_tokens: int


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)


def _nonblank(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunnerInputError(code)
    return value.strip()


def _safe_name(value: object, code: str) -> str:
    result = _nonblank(value, code)
    if _SAFE_NAME.fullmatch(result) is None:
        raise RunnerInputError(code)
    return result


def _sha256(value: object, code: str) -> str:
    result = _nonblank(value, code).lower()
    if _HEX_64.fullmatch(result) is None:
        raise RunnerInputError(code)
    return result


def _positive_int(value: object, code: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise RunnerInputError(code)
    return value


def _root_path(value: object, code: str) -> Path:
    return Path(_nonblank(value, code)).expanduser()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _index_path(index_root: Path, index_name: str) -> Path:
    return index_root / index_name


def _request() -> dict[str, Any]:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        raise RunnerInputError("invalid_json") from None
    if not isinstance(payload, dict):
        raise RunnerInputError("invalid_request")
    allowed = {
        "index_root", "index_name", "question", "top_k", "mode", "device",
        "generation_model", "generation_top_k", "max_tokens",
    }
    required = {"index_root", "index_name", "question", "top_k", "mode", "device"}
    if set(payload).difference(allowed) or not required.issubset(payload):
        raise RunnerInputError("invalid_request")
    return payload


def _read_json(path: Path, code: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RunnerInputError(code) from None


def _manifest(index_path: Path, *, expected_index_name: str | None = None) -> IndexManifest:
    manifest_path = index_path / _INDEX_MANIFEST
    raw = _read_json(manifest_path, "index_manifest_invalid")
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version", "index_name", "document_id", "source_id", "source_sha256",
        "page_count", "embedding_model", "index_backend", "created_at",
    }:
        raise RunnerInputError("index_manifest_invalid")
    manifest = IndexManifest(
        index_name=_safe_name(raw.get("index_name"), "index_manifest_invalid"),
        document_id=_nonblank(raw.get("document_id"), "index_manifest_invalid"),
        source_id=_nonblank(raw.get("source_id"), "index_manifest_invalid"),
        source_sha256=_sha256(raw.get("source_sha256"), "index_manifest_invalid"),
        page_count=_positive_int(raw.get("page_count"), "index_manifest_invalid", maximum=20_000),
        embedding_model=_nonblank(raw.get("embedding_model"), "index_manifest_invalid"),
        index_backend=_nonblank(raw.get("index_backend"), "index_manifest_invalid"),
        manifest_sha256=_file_sha256(manifest_path),
    )
    if raw.get("schema_version") != INDEX_SCHEMA or manifest.index_name != (
        expected_index_name or index_path.name
    ):
        raise RunnerInputError("index_manifest_invalid")
    if manifest.index_backend != "faiss":
        raise RunnerInputError("index_manifest_invalid")
    return manifest


def _valid_int(value: object, *, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _validate_index_layout(index_path: Path, manifest: IndexManifest) -> None:
    """Require exactly the single-PDF, one-vector-per-physical-page layout."""
    required = ("articles.json", "summary.json", "metadata.npz", "index.faiss")
    if not all((index_path / name).is_file() for name in required):
        raise RunnerInputError("index_not_found")
    articles = _read_json(index_path / "articles.json", "index_data_invalid")
    summary = _read_json(index_path / "summary.json", "index_data_invalid")
    if not isinstance(articles, list) or len(articles) != 1 or not isinstance(articles[0], Mapping):
        raise RunnerInputError("index_data_invalid")
    if not isinstance(summary, Mapping) or summary.get("backend") != "faiss":
        raise RunnerInputError("index_data_invalid")
    try:
        metadata = np.load(index_path / "metadata.npz", allow_pickle=False)
        names = {"article_ids", "tile_indices", "chunk_indices", "y_offsets", "tile_heights"}
        if set(metadata.files) != names:
            raise RunnerInputError("index_data_invalid")
        fields = {name: metadata[name] for name in names}
    except (OSError, ValueError, KeyError):
        raise RunnerInputError("index_data_invalid") from None
    finally:
        try:
            metadata.close()
        except UnboundLocalError:
            pass
    if any(values.ndim != 1 or len(values) != manifest.page_count for values in fields.values()):
        raise RunnerInputError("index_data_invalid")
    rows = sorted(
        zip(
            fields["article_ids"].tolist(),
            fields["tile_indices"].tolist(),
            fields["chunk_indices"].tolist(),
            fields["y_offsets"].tolist(),
            fields["tile_heights"].tolist(),
            strict=True,
        ),
        key=lambda row: row[1],
    )
    for page_index, (article_id, tile_index, chunk_index, y_offset, tile_height) in enumerate(rows):
        if (
            not _valid_int(article_id) or article_id != 0
            or not _valid_int(tile_index) or tile_index != page_index
            or not _valid_int(chunk_index) or chunk_index != 0
            or not _valid_int(y_offset) or y_offset != 0
            or not _valid_int(tile_height, minimum=1)
        ):
            raise RunnerInputError("index_data_invalid")
    tile_dir = index_path / "tiles" / "0.png.tiles"
    tiles = _read_json(tile_dir / "tiles.json", "index_data_invalid")
    chunks = _read_json(tile_dir / "chunks.json", "index_data_invalid")
    if not isinstance(tiles, Mapping) or not isinstance(chunks, Mapping):
        raise RunnerInputError("index_data_invalid")
    if (
        tiles.get("article_id") != 0
        or tiles.get("complete") is not True
        or tiles.get("total_pages") != manifest.page_count
        or not isinstance(tiles.get("source"), str)
        or chunks.get("article_id") != 0
        or chunks.get("num_chunks") != manifest.page_count
        or not isinstance(chunks.get("chunks"), list)
        or len(chunks["chunks"]) != manifest.page_count
    ):
        raise RunnerInputError("index_data_invalid")
    for page_index, item in enumerate(chunks["chunks"]):
        if not isinstance(item, Mapping) or item.get("tile_index") != page_index or item.get("chunk_index") != 0:
            raise RunnerInputError("index_data_invalid")
        image = tile_dir / f"tile_{page_index:04d}.jpg"
        if not image.is_file():
            raise RunnerInputError("index_data_invalid")


def _load_index(index_path: Path, *, expected_index_name: str | None = None) -> IndexManifest:
    manifest = _manifest(index_path, expected_index_name=expected_index_name)
    _validate_index_layout(index_path, manifest)
    return manifest


class _PixelRAGSearcher:
    """Direct in-process use of PixelRAG's local FAISS + embedding implementation."""

    def __init__(self, *, index_path: Path, model: str, device: str) -> None:
        # Both variables force HF/transformers into the already configured local
        # cache.  A query must fail rather than fetch a model or send document data.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        with contextlib.redirect_stdout(sys.stderr):
            from pixelrag_serve import api

            api.load(
                SimpleNamespace(
                    index_dir=str(index_path),
                    tiles_dir=str(index_path / "tiles"),
                    articles_json=str(index_path / "articles.json"),
                    backend=None,
                    qdrant_url=None,
                    qdrant_api_key=None,
                    qdrant_client_config=None,
                    collection=None,
                    model=model,
                    device=device,
                    peft_adapter=None,
                    render_on_demand=False,
                    kiwix_url=None,
                    zim_book=None,
                )
            )
        self._api = api

    def search(self, question: str, limit: int) -> list[Mapping[str, object]]:
        with contextlib.redirect_stdout(sys.stderr):
            vectors = self._api._encode_queries([self._api.Query(text=question)])
            return list(self._api._state["backend"].raw_search(vectors, limit)[0])

    def close(self) -> None:
        state = self._api._state
        state.clear()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def _searcher_factory(*, index_path: Path, model: str, device: str) -> Searcher:
    return _PixelRAGSearcher(index_path=index_path, model=model, device=device)


def _rank_pages(
    *,
    hits: Sequence[Mapping[str, object]],
    manifest: IndexManifest,
    top_k: int,
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    seen_pages: set[int] = set()
    for hit in hits:
        if not isinstance(hit, Mapping):
            raise RunnerInputError("search_result_invalid")
        expected = {"score", "vector_id", "article_id", "tile_index", "chunk_index", "y_offset", "tile_height"}
        if set(hit) != expected:
            raise RunnerInputError("search_result_invalid")
        score = hit.get("score")
        page_index = hit.get("tile_index")
        if (
            isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not _valid_int(hit.get("article_id")) or hit["article_id"] != 0
            or not _valid_int(page_index) or page_index >= manifest.page_count
            or not _valid_int(hit.get("chunk_index")) or hit["chunk_index"] != 0
            or not _valid_int(hit.get("y_offset")) or hit["y_offset"] != 0
            or not _valid_int(hit.get("tile_height"), minimum=1)
        ):
            raise RunnerInputError("search_result_invalid")
        if page_index in seen_pages:
            continue
        seen_pages.add(page_index)
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "document_id": manifest.document_id,
                "source_id": manifest.source_id,
                "pdf_page_index": page_index,
                "native_page": page_index + 1,
                "score": float(score),
                "article_id": 0,
                "tile_index": page_index,
                "chunk_index": 0,
                "y_offset": 0,
                "tile_height": int(hit["tile_height"]),
                "image": f"tiles/0.png.tiles/tile_{page_index:04d}.jpg",
            }
        )
        if len(ranked) == top_k:
            break
    return ranked


def _generation_config(payload: Mapping[str, object], *, top_k: int) -> GenerationConfig:
    for key in ("generation_model", "generation_top_k", "max_tokens"):
        if key not in payload:
            raise RunnerInputError("generation_config_missing")
    model = _root_path(payload.get("generation_model"), "generation_model_missing")
    if not model.is_dir():
        raise RunnerInputError("generation_model_not_local")
    image_count = _positive_int(payload.get("generation_top_k"), "generation_top_k_invalid", maximum=5)
    if image_count > top_k:
        raise RunnerInputError("generation_top_k_invalid")
    return GenerationConfig(
        model=model,
        image_count=image_count,
        max_tokens=_positive_int(payload.get("max_tokens"), "max_tokens_invalid", maximum=4096),
    )


def _strip_reasoning(value: str) -> str:
    return re.sub(r"<think>.*?</think>", "", value, flags=re.IGNORECASE | re.DOTALL).strip()


def _generate_answer(*, question: str, image_paths: Sequence[Path], config: GenerationConfig, device: str) -> str:
    """Run an already-cached local VLM; no hosted reader or provider is used."""
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise RunnerInputError("generation_runtime_unavailable") from error
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with contextlib.redirect_stdout(sys.stderr):
            processor = AutoProcessor.from_pretrained(
                str(config.model), trust_remote_code=True, local_files_only=True
            )
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(config.model),
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            ).eval().to(device)
            prompt = (
                "Answer the question using only the provided document-page images. "
                f"If the evidence is insufficient, respond with exactly {_ABSTENTION}. "
                "Do not reveal reasoning, analysis, or <think> tags.\n\n"
                f"Question: {question}"
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *[
                {"type": "image", "image": image} for image in images
            ]]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=config.max_tokens, do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            answer = processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0]
    finally:
        for image in images:
            image.close()
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    answer = _strip_reasoning(answer)
    if not answer:
        raise RunnerInputError("generation_empty")
    return answer


def _identity(manifest: IndexManifest) -> dict[str, object]:
    return {
        "index_name": manifest.index_name,
        "document_id": manifest.document_id,
        "source_id": manifest.source_id,
        "source_sha256": manifest.source_sha256,
        "page_count": manifest.page_count,
        "embedding_model": manifest.embedding_model,
        "index_backend": manifest.index_backend,
        "index_manifest_sha256": manifest.manifest_sha256,
    }


def _query(
    payload: Mapping[str, object],
    *,
    searcher_factory: Callable[..., Searcher] = _searcher_factory,
    generator: Callable[..., str] = _generate_answer,
) -> dict[str, object]:
    index_root = _root_path(payload.get("index_root"), "index_root_missing")
    index_name = _safe_name(payload.get("index_name"), "index_name_missing")
    question = _nonblank(payload.get("question"), "question_missing")
    top_k = _positive_int(payload.get("top_k"), "top_k_invalid", maximum=100)
    device = _nonblank(payload.get("device"), "device_missing")
    mode = payload.get("mode")
    if mode not in {"retriever_only", "full_rag"}:
        raise RunnerInputError("mode_unsupported")
    manifest = _load_index(_index_path(index_root, index_name))
    searcher: Searcher | None = None
    try:
        searcher = searcher_factory(
            index_path=_index_path(index_root, index_name), model=manifest.embedding_model, device=device
        )
        # The validated PDF profile has exactly one vector per page.  Fetch all
        # pages, then preserve the FAISS ranking while applying the page mapping.
        ranked = _rank_pages(
            hits=searcher.search(question, manifest.page_count), manifest=manifest, top_k=top_k
        )
    finally:
        if searcher is not None:
            searcher.close()
    response: dict[str, object] = {
        "schema_version": RUNNER_SCHEMA,
        "outcome": "retrieval_only",
        "answer": None,
        "results": ranked,
        "generation": {"requested": False, "model": None, "evidence_ranks": []},
        "identity": _identity(manifest),
    }
    if mode == "retriever_only":
        return response
    if not ranked:
        response.update(outcome="abstained")
        response["generation"] = {"requested": True, "model": None, "evidence_ranks": []}
        return response
    config = _generation_config(payload, top_k=len(ranked))
    selected = ranked[:config.image_count]
    index_path = _index_path(index_root, index_name)
    image_paths = [index_path / str(row["image"]) for row in selected]
    if not all(path.is_file() for path in image_paths):
        raise RunnerInputError("index_data_invalid")
    generated = generator(question=question, image_paths=image_paths, config=config, device=device)
    answer = _strip_reasoning(generated)
    response.update(outcome="abstained" if answer.casefold() == _ABSTENTION.casefold() else "answered")
    response["answer"] = None if answer.casefold() == _ABSTENTION.casefold() else answer
    response["generation"] = {
        "requested": True,
        "model": str(config.model),
        "evidence_ranks": [int(row["rank"]) for row in selected],
    }
    return response


def _atomic_replace(build_path: Path, target: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    try:
        if target.exists():
            if not overwrite:
                raise RunnerInputError("index_already_exists")
            backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
            target.replace(backup)
        build_path.replace(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.replace(target)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


def _build_index(
    *,
    source: Path,
    index_root: Path,
    index_name: str,
    document_id: str,
    source_id: str,
    source_sha256: str,
    embedding_model: str,
    device: str,
    overwrite: bool,
    builder: Callable[[dict[str, object]], Path] | None = None,
) -> dict[str, object]:
    """Explicitly build and atomically publish the narrow evaluation index."""
    if not source.is_file():
        raise RunnerInputError("input_document_missing")
    if _file_sha256(source) != source_sha256:
        raise RunnerInputError("input_document_checksum_mismatch")
    index_root.mkdir(parents=True, exist_ok=True)
    target = _index_path(index_root, index_name)
    if target.exists() and not overwrite:
        raise RunnerInputError("index_already_exists")
    build_path = Path(tempfile.mkdtemp(prefix=f".{index_name}.build-", dir=index_root))
    try:
        config: dict[str, object] = {
            "source": {"type": "pdf", "path": str(source)},
            "output": str(build_path),
            # Use PixelRAG's direct transformers embedder on CUDA rather than
            # its optional vLLM/SGLang path.  This remains in-process/local and
            # is compatible with one complete PDF page per visual vector.
            "embed": {"model": embedding_model, "device": device, "backend": "transformers"},
            "index": {"backend": "faiss"},
        }
        if builder is None:
            from pixelrag_index.pipelines import build

            build(config, force=False)
        else:
            built = builder(config)
            if Path(built) != build_path:
                raise RunnerInputError("index_build_invalid")
        tiles = _read_json(build_path / "tiles" / "0.png.tiles" / "tiles.json", "index_build_invalid")
        if not isinstance(tiles, Mapping):
            raise RunnerInputError("index_build_invalid")
        page_count = _positive_int(tiles.get("total_pages"), "index_build_invalid", maximum=20_000)
        manifest = {
            "schema_version": INDEX_SCHEMA,
            "index_name": index_name,
            "document_id": document_id,
            "source_id": source_id,
            "source_sha256": source_sha256,
            "page_count": page_count,
            "embedding_model": embedding_model,
            "index_backend": "faiss",
            "created_at": _utc_now(),
        }
        (build_path / _INDEX_MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        _load_index(build_path, expected_index_name=index_name)
        _atomic_replace(build_path, target, overwrite=overwrite)
    except Exception:
        if build_path.exists():
            shutil.rmtree(build_path, ignore_errors=True)
        raise
    return {
        "schema_version": RUNNER_SCHEMA,
        "outcome": "indexed",
        "index_name": index_name,
        "document_id": document_id,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "page_count": page_count,
        "index_schema": INDEX_SCHEMA,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("query", help="Read one JSON request from stdin and emit one JSON response")
    index = subcommands.add_parser("index", help="Build one explicit local PDF index")
    index.add_argument("--input", required=True)
    index.add_argument("--index-root", required=True)
    index.add_argument("--index-name", required=True)
    index.add_argument("--document-id", required=True)
    index.add_argument("--source-id", required=True)
    index.add_argument("--source-sha256", required=True)
    index.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    index.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"])
    index.add_argument("--overwrite", action="store_true")
    return parser


def _index(arguments: argparse.Namespace) -> dict[str, object]:
    return _build_index(
        source=Path(arguments.input).expanduser(),
        index_root=Path(arguments.index_root).expanduser(),
        index_name=_safe_name(arguments.index_name, "index_name_invalid"),
        document_id=_nonblank(arguments.document_id, "document_id_missing"),
        source_id=_nonblank(arguments.source_id, "source_id_missing"),
        source_sha256=_sha256(arguments.source_sha256, "source_sha256_invalid"),
        embedding_model=_nonblank(arguments.embedding_model, "embedding_model_missing"),
        device=_nonblank(arguments.device, "device_missing"),
        overwrite=arguments.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        response = _query(_request()) if arguments.command == "query" else _index(arguments)
    except RunnerInputError as error:
        _emit({"schema_version": RUNNER_SCHEMA, "outcome": "error", "error_code": str(error)})
        return 2
    except Exception:
        _emit({"schema_version": RUNNER_SCHEMA, "outcome": "error", "error_code": "runner_failed"})
        return 1
    _emit(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
