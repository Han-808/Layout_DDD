from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.assets.retriever import AssetRetriever, build_asset_index_from_asset_info


class _FakeEmbeddingModel:
    def __init__(self, embeddings: dict[str, list[float]] | None = None) -> None:
        self.embeddings = embeddings or {}
        self.calls: list[tuple[str, dict]] = []

    def encode(self, text: str, **kwargs) -> np.ndarray:
        self.calls.append((text, kwargs))
        return np.asarray(self.embeddings.get(text, [1.0, 0.0]), dtype=np.float32)


def test_encode_text_matches_intelliscene_query_document_protocol() -> None:
    model = _FakeEmbeddingModel()
    retriever = AssetRetriever(device="cpu")
    retriever._model = model

    retriever.encode_text("a red bed", is_query=True)
    retriever.encode_text("red upholstered bed", is_query=False)

    assert model.calls == [
        ("a red bed", {"prompt_name": "query", "convert_to_numpy": True}),
        ("red upholstered bed", {"convert_to_numpy": True}),
    ]


def test_query_prompt_errors_are_not_silently_downgraded() -> None:
    class UnsupportedQueryPromptModel:
        def encode(self, text: str, **kwargs) -> np.ndarray:
            if "prompt_name" in kwargs:
                raise TypeError("prompt_name is unsupported")
            return np.asarray([1.0, 0.0], dtype=np.float32)

    retriever = AssetRetriever(device="cpu")
    retriever._model = UnsupportedQueryPromptModel()

    with pytest.raises(TypeError, match="prompt_name"):
        retriever.encode_text("a red bed", is_query=True)


def test_retrieve_matches_intelliscene_query_and_fallback_behavior() -> None:
    model = _FakeEmbeddingModel({"bed a red bed": [0.1, 0.2]})
    retriever = AssetRetriever(device="cpu")
    retriever._model = model
    retriever.index.assets = {
        "asset_a": {
            "jid": "asset_a",
            "short_desc": "red bed",
            "size": [1.0, 1.0, 1.0],
            "category": "bed",
            "description": "red bed",
        },
        "asset_b": {
            "jid": "asset_b",
            "short_desc": "blue bed",
            "size": [1.0, 1.0, 1.0],
            "category": "bed",
            "description": "blue bed",
        },
    }
    retriever.index.jid_list = ["asset_a", "asset_b"]
    retriever.index.embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    results = retriever.retrieve(
        description="a red bed",
        category="bed",
        top_k=5,
        min_score=0.99,
    )

    assert model.calls == [
        ("bed a red bed", {"prompt_name": "query", "convert_to_numpy": True})
    ]
    assert [result["jid"] for result in results] == ["asset_b"]


def test_index_builder_uses_short_desc_and_metadata_size(tmp_path, monkeypatch) -> None:
    asset_root = tmp_path / "imaginarium_assets"
    asset_dir = asset_root / "asset_001"
    asset_dir.mkdir(parents=True)
    (asset_dir / "asset_001_metadata.json").write_text(
        json.dumps({"transformed_size": [2.1, 1.2, 0.8]}),
        encoding="utf-8",
    )
    csv_path = asset_root / "imaginarium_asset_info.csv"
    csv_path.write_text(
        "id,name_en,bbx,caption_en,short_desc,category\n"
        '1,asset_001,"2.0,1.0,0.7",A detailed red upholstered bed,red upholstered bed,bed\n',
        encoding="utf-8",
    )
    model = _FakeEmbeddingModel({"red upholstered bed": [0.25, 0.75]})
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda model_name: model),
    )

    index = build_asset_index_from_asset_info(
        asset_info_csv_path=csv_path,
        asset_dir=asset_root,
        output_path=tmp_path / "asset_index",
    )

    assert model.calls == [
        ("red upholstered bed", {"convert_to_numpy": True})
    ]
    assert index.assets["asset_001"]["size"] == [2.1, 1.2, 0.8]
    assert index.assets["asset_001"]["description"] == "A detailed red upholstered bed"
    np.testing.assert_array_equal(index.embeddings, np.asarray([[0.25, 0.75]], dtype=np.float32))
    assert (tmp_path / "asset_index.json").is_file()
    assert (tmp_path / "asset_index.npy").is_file()
