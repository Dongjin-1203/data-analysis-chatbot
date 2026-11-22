"""OpenAI 호출 없이 단어 기반 가짜 임베딩으로 매뉴얼 검색 흐름을 검증한다."""

import hashlib
import re

import numpy as np

from app.rag import MANUAL_DIR, ManualIndex, load_chunks

DIM = 512


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        v = np.zeros(DIM)
        for token in re.findall(r"[A-Za-z_]+|[가-힣]+", text):
            v[int(hashlib.md5(token.lower().encode()).hexdigest(), 16) % DIM] += 1
        vectors.append(v.tolist())
    return vectors


def test_manuals_are_chunked_by_section():
    chunks = load_chunks()
    docs = {c.doc_id for c in chunks}
    assert docs == {f"MAN-0{i}" for i in range(1, 7)}
    assert all(c.section and c.text.startswith("[") for c in chunks)
    assert any(c.doc_id == "MAN-03" and "점검 절차" in c.section for c in chunks)
    # 모든 매뉴얼은 가상 문서임을 밝힌다
    for path in MANUAL_DIR.glob("*.md"):
        assert "가상 매뉴얼" in path.read_text(encoding="utf-8")


def test_search_finds_relevant_manual(tmp_path):
    index = ManualIndex(embed_fn=fake_embed, cache_path=tmp_path / "emb.json")
    hits = index.search("PRESS_CHAMBER_DEP1_S 챔버 압력 진공 누설 점검", k=3)
    assert hits[0][0].doc_id == "MAN-03"
    assert hits[0][1] >= hits[-1][1]
    assert hits[0][0].citation.startswith("[MAN-03 ")


def test_embeddings_are_cached(tmp_path):
    calls = []

    def counting_embed(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    cache = tmp_path / "emb.json"
    ManualIndex(embed_fn=counting_embed, cache_path=cache).search("타깃 파워", k=2)
    n_chunks = len(load_chunks())
    assert calls == [n_chunks, 1]  # 청크 전체 + 질문

    calls.clear()
    ManualIndex(embed_fn=counting_embed, cache_path=cache).search("플래튼 온도", k=2)
    assert calls == [1]  # 새 인스턴스도 캐시를 써서 질문만 임베딩
