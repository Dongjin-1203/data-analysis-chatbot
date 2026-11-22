"""점검 매뉴얼 검색 (RAG).

- manuals/*.md 를 '## ' 절 단위로 나눠 청크를 만든다.
- 청크와 질문을 text-embedding-3-small 로 임베딩하고 코사인 유사도로 검색한다.
- 청크 임베딩은 storage/manual_embeddings.json 에 내용 해시로 캐시한다(매뉴얼을 고치면 그 청크만 다시 임베딩).
"""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import get_settings

MANUAL_DIR = Path(__file__).resolve().parent.parent / "manuals"

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str
    file: str

    @property
    def citation(self) -> str:
        return f"[{self.doc_id} {self.section}]"

    @property
    def key(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_chunks(manual_dir: Path = MANUAL_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(manual_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        id_match = re.search(r"\[([A-Z]+-\d+)\]", title)
        doc_id = id_match.group(1) if id_match else path.stem
        title_only = re.sub(r"^\[[^\]]+\]\s*", "", title)
        for part in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
            heading, _, body = part.partition("\n")
            body = body.strip()
            if not body:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    title=title_only,
                    section=heading.strip(),
                    text=f"{title}\n## {heading.strip()}\n{body}",
                    file=path.name,
                )
            )
    return chunks


def openai_embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않아 매뉴얼을 검색할 수 없습니다.")
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


class ManualIndex:
    def __init__(self, embed_fn: EmbedFn | None = None, cache_path: Path | None = None, manual_dir: Path = MANUAL_DIR):
        self.embed_fn = embed_fn or openai_embed
        self.cache_path = cache_path or Path(get_settings().storage_dir).resolve() / "manual_embeddings.json"
        self.manual_dir = manual_dir
        self.model = get_settings().embedding_model
        self._cache: dict[str, list[float]] | None = None

    def _load_cache(self) -> dict[str, list[float]]:
        if self._cache is None:
            self._cache = {}
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if data.get("model") == self.model:
                    self._cache = data.get("items", {})
        return self._cache

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({"model": self.model, "items": self._cache}), encoding="utf-8")

    def embeddings(self, chunks: list[Chunk]) -> np.ndarray:
        cache = self._load_cache()
        todo = [c for c in chunks if c.key not in cache]
        if todo:
            for c, vec in zip(todo, self.embed_fn([c.text for c in todo])):
                cache[c.key] = vec
            self._save_cache()
        return np.array([cache[c.key] for c in chunks], dtype=np.float32)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        chunks = load_chunks(self.manual_dir)
        if not chunks:
            return []
        matrix = self.embeddings(chunks)
        q = np.array(self.embed_fn([query])[0], dtype=np.float32)
        sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9)
        order = np.argsort(-sims)[: max(1, min(k, 8))]
        return [(chunks[i], float(sims[i])) for i in order]


_index: ManualIndex | None = None


def get_index() -> ManualIndex:
    global _index
    if _index is None:
        _index = ManualIndex()
    return _index
