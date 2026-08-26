import numpy as np
import torch

import rag_agent


def test_tfidf_index_is_reused_and_invalidated(tmp_path):
    csv_path = tmp_path / "legal_stats.csv"
    csv_path.write_text(
        "FileName,TextContent,CrimeType\n"
        "a.pdf,詐欺取財案件,詐欺\n"
        "b.pdf,竊盜案件事實,竊盜\n",
        encoding="utf-8",
    )
    rag_agent.clear_runtime_caches()

    first = rag_agent._get_tfidf_index(csv_path, np.array([0, 1]))
    second = rag_agent._get_tfidf_index(csv_path, np.array([0, 1]))

    assert second is first

    csv_path.write_text(
        "FileName,TextContent,CrimeType\n"
        "a.pdf,詐欺取財案件,詐欺\n"
        "b.pdf,竊盜案件事實,竊盜\n"
        "c.pdf,洗錢案件內容新增,洗錢\n",
        encoding="utf-8",
    )
    refreshed = rag_agent._get_tfidf_index(csv_path, np.array([0, 1, 2]))

    assert refreshed is not first
    assert len(refreshed["texts"]) == 3


def test_latent_feature_cache_avoids_repeated_encoding(monkeypatch):
    class DummyVocab:
        def encode(self, _text, max_len):
            return [2] + [0] * (max_len - 1)

    class DummyModel:
        def __init__(self):
            self.calls = 0

        def encode(self, _inputs):
            self.calls += 1
            return torch.tensor([[1.0, 2.0, 3.0]])

    model = DummyModel()
    monkeypatch.setattr(rag_agent, "_model", model)
    monkeypatch.setattr(rag_agent, "_vocab", DummyVocab())
    monkeypatch.setattr(rag_agent, "_model_signature", ("test-model", 1, 1))
    monkeypatch.setattr(rag_agent.config, "TRAIN_MAX_SEQ_LEN", 4)
    monkeypatch.setattr(rag_agent.config, "RAG_FEATURE_CACHE_SIZE", 2)
    rag_agent.clear_runtime_caches()

    first = rag_agent.get_latent_features("相同文本")
    second = rag_agent.get_latent_features("相同文本")

    assert model.calls == 1
    assert np.array_equal(first, second)

    first[0] = 99.0
    third = rag_agent.get_latent_features("相同文本")
    assert third[0] == 1.0
