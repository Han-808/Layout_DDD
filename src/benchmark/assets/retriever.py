from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_SIZE_SCORE_CLOSE = 0.08
_SIZE_SCORE_MODERATE = 0.03
_SIZE_PENALTY_FAR = -0.02
_SIZE_PENALTY_VERY_FAR = -0.08


def _compute_size_score(asset_size: list[float] | None, target_size: list[float] | None, tolerance: float) -> float:
    """Return a small soft score based on logarithmic size distance."""

    if not asset_size or not target_size or len(asset_size) != 3 or len(target_size) != 3:
        return 0.0
    valid_diffs = []
    for asset_dim, target_dim in zip(asset_size, target_size):
        if target_dim <= 0 or asset_dim <= 0:
            continue
        valid_diffs.append(abs(np.log(float(asset_dim) / float(target_dim))))
    if not valid_diffs:
        return 0.0
    mean_diff = float(np.mean(valid_diffs))
    if mean_diff <= np.log(1 + tolerance):
        return _SIZE_SCORE_CLOSE
    if mean_diff <= np.log(1 + tolerance * 2):
        return _SIZE_SCORE_MODERATE
    if mean_diff <= np.log(1 + tolerance * 3):
        return _SIZE_PENALTY_FAR
    return _SIZE_PENALTY_VERY_FAR


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _parse_size_field(value: Any) -> list[float] | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == 3:
            size = [float(item) for item in parsed]
            if all(dim > 0 for dim in size):
                return size
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        size = [float(item.strip()) for item in text.split(",")]
        if len(size) == 3 and all(dim > 0 for dim in size):
            return size
    except ValueError:
        pass
    return None


class AssetIndex:
    """Small serializable asset metadata + embedding index."""

    def __init__(self) -> None:
        self.assets: dict[str, dict[str, Any]] = {}
        self.embeddings: np.ndarray | None = None
        self.jid_list: list[str] = []
        self._embedding_buffer: list[np.ndarray] = []

    def add_asset(
        self,
        jid: str,
        short_desc: str,
        size: list[float],
        *,
        category: str = "",
        description: str = "",
        embedding: np.ndarray | None = None,
    ) -> None:
        self.assets[jid] = {
            "jid": jid,
            "short_desc": short_desc,
            "size": size,
            "category": category,
            "description": description or short_desc,
        }
        if jid not in self.jid_list:
            self.jid_list.append(jid)
            if embedding is not None:
                self._embedding_buffer.append(embedding.reshape(1, -1))

    def finalize(self) -> None:
        if self._embedding_buffer:
            self.embeddings = np.vstack(self._embedding_buffer)
            self._embedding_buffer.clear()

    def get_asset(self, jid: str) -> dict[str, Any] | None:
        return self.assets.get(jid)

    def save(self, path: str | Path) -> None:
        self.finalize()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump({"assets": self.assets, "jid_list": self.jid_list}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if self.embeddings is not None:
            np.save(save_path.with_suffix(".npy"), self.embeddings)

    def load(self, path: str | Path) -> None:
        load_path = Path(path)
        with load_path.with_suffix(".json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.assets = data["assets"]
        self.jid_list = data["jid_list"]
        embeddings_path = load_path.with_suffix(".npy")
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)

    def __len__(self) -> int:
        return len(self.assets)

    def __contains__(self, jid: str) -> bool:
        return jid in self.assets


class AssetRetriever:
    """Retrieve assets by semantic text similarity plus optional size soft scoring."""

    def __init__(
        self,
        index_path: str | Path | None = None,
        *,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str | None = None,
        use_flash_attention: bool = False,
    ) -> None:
        self.index = AssetIndex()
        self.embedding_model_name = embedding_model
        self.device = device or _default_device()
        self.use_flash_attention = use_flash_attention
        self._model: Any = None
        if index_path:
            self.index.load(index_path)

    def _get_embedding_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional runtime dependency
                raise RuntimeError("sentence-transformers is required to encode retrieval queries.") from exc
            if self.use_flash_attention and "cuda" in str(self.device):
                import torch

                self._model = SentenceTransformer(
                    self.embedding_model_name,
                    model_kwargs={
                        "attn_implementation": "flash_attention_2",
                        "torch_dtype": torch.bfloat16,
                        "device_map": self.device,
                    },
                    tokenizer_kwargs={"padding_side": "left"},
                )
            else:
                self._model = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._model

    def encode_text(self, text: str, *, is_query: bool = True) -> np.ndarray:
        model = self._get_embedding_model()
        if is_query:
            try:
                return model.encode(text, prompt_name="query", convert_to_numpy=True)
            except TypeError:
                return model.encode(text, convert_to_numpy=True)
        return model.encode(text, convert_to_numpy=True)

    def retrieve(
        self,
        description: str,
        *,
        category: str | None = None,
        size_constraint: list[float] | None = None,
        size_tolerance: float = 0.5,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        if len(self.index) == 0:
            return []
        if self.index.embeddings is None:
            raise ValueError("Asset index does not contain embeddings. Please rebuild the index with embeddings.")
        query_text = f"{category} {description}".strip() if category else description
        query_embedding = self.encode_text(query_text, is_query=True)
        norms = np.linalg.norm(self.index.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        cosine_scores = self.index.embeddings @ query_embedding / norms
        scored: list[dict[str, Any]] = []
        for index, jid in enumerate(self.index.jid_list):
            asset = self.index.assets[jid]
            score = float(cosine_scores[index])
            score += _compute_size_score(asset.get("size"), size_constraint, size_tolerance)
            scored.append({**asset, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        results = [item for item in scored if item["score"] >= min_score]
        return (results or scored)[: max(1, int(top_k))]


def build_asset_index_from_asset_info(
    asset_info_csv_path: str | Path,
    asset_dir: str | Path,
    output_path: str | Path,
    *,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
) -> AssetIndex:
    """Build an asset index from an Imaginarium-style asset_info CSV."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("sentence-transformers is required to build an asset index.") from exc
    index = AssetIndex()
    model = SentenceTransformer(embedding_model)
    asset_root = Path(asset_dir)
    with Path(asset_info_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            jid = _clean_text(row.get("jid") or row.get("name_en"))
            if not jid:
                continue
            asset_path = asset_root / jid
            if not asset_path.exists():
                continue
            short_desc = _clean_text(row.get("short_desc"))
            description = _clean_text(row.get("desc") or row.get("caption_en")) or short_desc
            category = _clean_text(row.get("category") or row.get("class_en")).lower().replace("_", " ")
            size = _parse_size_field(row.get("size") or row.get("bbx"))
            metadata_path = asset_path / f"{jid}_metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    size = metadata.get("transformed_size") or size
                except (OSError, json.JSONDecodeError):
                    pass
            if not short_desc:
                raise ValueError(f"Asset {jid} is missing short_desc in asset info CSV.")
            if not size or len(size) != 3:
                raise ValueError(f"Asset {jid} is missing valid size in asset info and metadata.")
            embedding = model.encode(short_desc, convert_to_numpy=True)
            index.add_asset(
                jid=jid,
                short_desc=short_desc,
                size=[max(float(item), 0.01) for item in size],
                category=category,
                description=description,
                embedding=embedding,
            )
    index.save(output_path)
    return index


def _default_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
