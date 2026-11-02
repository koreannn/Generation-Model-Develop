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


if __name__ == "__main__":
    build()
