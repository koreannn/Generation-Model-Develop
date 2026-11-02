"""
청크 코퍼스 BM25 인덱스 생성 (bm25s + Kiwi 형태소 분석)

- 입력: data/wiki/kowiki_chunks.jsonl (chunk_wiki_corpus.py 출력)
- 출력: data/wiki/bm25_index/ (bm25s 인덱스 + 행 번호 → chunk_id 목록)
- 토크나이저와 검색(BM25Retriever)은 code/rag/bm25_search.py에 있음
  인덱싱과 검색이 같은 토크나이저를 써야 하므로 여기서는 가져다 쓰기만 함

사용법
    uv run python data_process/bm25_indexer.py   # 인덱스 생성 (약 15분)
"""

import json
from pathlib import Path
import sys

import bm25s
from loguru import logger
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "rag"))
from bm25_search import CHUNKS_PATH, INDEX_DIR, get_kiwi, to_terms  # noqa: E402


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
