"""
청크 코퍼스 BM25 인덱스 (bm25s + Kiwi 형태소 분석)

- 입력: data/wiki/kowiki_chunks.jsonl (chunk_wiki_corpus.py 출력)
- 출력: data/wiki/bm25_index/ (bm25s 인덱스 + 행 번호 → chunk_id 목록)
- 토큰: Kiwi 형태소 중 내용어만 사용 (명사, 숫자, 외국어, 한자, 어근, 동사·형용사 어간)
  조사·어미가 섞이면 "원자론을"과 "원자론은"이 다른 토큰이 되어 BM25가 매칭하지 못함
- 정답 근거 레이블링의 후보 수집(build_gold_candidates.py)에서 사용
    - search: 전체 코퍼스에서 상위 k개 청크
    - rank_in_doc: 특정 문서의 청크만 같은 쿼리 점수로 정렬

사용법
    uv run python data_process/bm25_index.py   # 인덱스 생성 (약 15분)
"""

import json
from pathlib import Path

import bm25s
from kiwipiepy import Kiwi
from loguru import logger
import numpy as np


WIKI_DIR = Path(__file__).resolve().parent.parent / "data" / "wiki"
CHUNKS_PATH = WIKI_DIR / "kowiki_chunks.jsonl"
INDEX_DIR = WIKI_DIR / "bm25_index"
CHUNK_IDS_PATH = INDEX_DIR / "chunk_ids.json"

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


def build(chunks_path=CHUNKS_PATH, index_dir=INDEX_DIR):
    chunk_ids = []

    def iter_texts():
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                chunk_ids.append(chunk["chunk_id"])
                yield chunk["text"]

    vocab = {}
    corpus_ids = []
    for tokens in get_kiwi().tokenize(iter_texts()):
        corpus_ids.append([vocab.setdefault(term, len(vocab)) for term in to_terms(tokens)])
        if len(corpus_ids) % 200000 == 0:
            logger.info(f"형태소 분석 {len(corpus_ids)}청크, 어휘 {len(vocab)}개")

    avg_terms = np.mean([len(c) for c in corpus_ids])
    logger.info(f"청크 {len(corpus_ids)}개, 어휘 {len(vocab)}개, 청크당 평균 {avg_terms:.0f}토큰")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenization.Tokenized(ids=corpus_ids, vocab=vocab), show_progress=False)
    retriever.save(index_dir, show_progress=False)
    with open(index_dir / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)
    logger.info(f"저장 완료: {index_dir}")


class BM25Index:
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


if __name__ == "__main__":
    build()
