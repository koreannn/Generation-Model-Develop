"""
청크 코퍼스 BM25 검색 (bm25s + Kiwi 형태소 분석)

- 인덱스: data/wiki/bm25_index/ (data_process/bm25_indexer.py로 생성)
- 토큰: Kiwi 형태소 중 내용어만 사용 (명사, 숫자, 외국어, 한자, 어근, 동사·형용사 어간)
  조사·어미가 섞이면 "원자론을"과 "원자론은"이 다른 토큰이 되어 BM25가 매칭하지 못함
  인덱싱과 검색이 같은 토크나이저를 써야 하므로 bm25_indexer.py도 여기의 tokenize를 가져다 씀
- BM25Retriever
    - search: 전체 코퍼스에서 상위 k개 청크
    - rank_in_doc: 특정 문서의 청크만 같은 쿼리 점수로 정렬 (정답 근거 레이블링용)
"""

import json
from pathlib import Path

import bm25s
from kiwipiepy import Kiwi
import numpy as np


WIKI_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "wiki"
CHUNKS_PATH = WIKI_DIR / "kowiki_chunks.jsonl"
INDEX_DIR = WIKI_DIR / "bm25_index"

# 일반명사, 고유명사, 수사, 숫자, 외국어, 한자, 어근, 동사, 형용사
KEEP_TAGS = {"NNG", "NNP", "NR", "SN", "SL", "SH", "XR", "VV", "VA"}

_kiwi = None


def get_kiwi():
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi(num_workers=-1)  # 모든 코어 사용
    return _kiwi


def to_terms(tokens):
    # 동사·형용사는 규칙/불규칙 활용에 따라 "VV-R", "VA-I"처럼 태그가 붙음
    return [t.form.lower() for t in tokens if t.tag.split("-")[0] in KEEP_TAGS]


def tokenize(texts):
    """문자열 리스트(또는 이터러블) → 토큰 리스트의 리스트"""
    return [to_terms(tokens) for tokens in get_kiwi().tokenize(texts)]


class BM25Retriever:
    def __init__(self, index_dir=INDEX_DIR):
        self.retriever = bm25s.BM25.load(index_dir, mmap=True)
        with open(index_dir / "chunk_ids.json", encoding="utf-8") as f:
            self.chunk_ids = json.load(f)
        self.doc_rows = {}
        for row, chunk_id in enumerate(self.chunk_ids):
            self.doc_rows.setdefault(chunk_id.rsplit("-", 1)[0], []).append(row)

    def scores(self, query):
        """쿼리 하나에 대한 전체 청크 점수 (어휘에 없는 토큰은 무시)"""
        terms = [t for t in tokenize([query])[0] if t in self.retriever.vocab_dict]
        if not terms:
            return np.zeros(len(self.chunk_ids), dtype=np.float32)
        return self.retriever.get_scores(terms)

    def search(self, query, k):
        """[(chunk_id, 점수), ...] 점수 내림차순, 점수 0인 청크 제외"""
        scores = self.scores(query)
        top = np.argpartition(-scores, k)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(self.chunk_ids[row], float(scores[row])) for row in top if scores[row] > 0]

    def rank_in_doc(self, scores, doc_id, k):
        """scores(전체 청크 점수) 기준으로 한 문서의 청크 상위 k개를 [(chunk_id, 점수), ...]로 반환"""
        rows = sorted(self.doc_rows.get(doc_id, []), key=lambda row: -scores[row])[:k]
        return [(self.chunk_ids[row], float(scores[row])) for row in rows]
