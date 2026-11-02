"""
검색기 평가 (Recall@k)

- 대상: oracle_eval_set.jsonl 중 config의 split, gold_status == gold인 문제
- 쿼리 방식 (정답을 모르는 상태에서 쓸 수 있는 정보만 사용)
    - question: 질문만
    - paragraph_question: 지문 + 질문
    - paragraph_question_choices: 지문 + 질문 + 선택지 전체 (검색 1번)
    - per_choice: "질문 + 선택지 i"로 선택지마다 검색한 뒤 청크별 최고 점수로 합침
- 지표 (k = 1, 3, 5, 10, 20)
    - chunk_any: 상위 k개에 근거 청크가 하나라도 있음 (메인)
    - chunk_all: 근거 청크가 모두 있음
    - doc_any: 상위 k개에 근거 문서(doc_id)가 하나라도 있음
- 레이블 후보를 모을 때 paragraph_question, per_choice와 같은 형태의 BM25 쿼리를 썼으므로 이 둘은 점수가 높게 나올 수 있음

사용법
    uv run python code/rag/retriever_eval.py
"""

import yaml
from loguru import logger

from bm25_search import BM25Retriever
from oracle_eval import CONFIG_PATH, load_eval_set


K_LIST = (1, 3, 5, 10, 20)
QUERY_TYPES = ("question", "paragraph_question", "paragraph_question_choices", "per_choice")


def build_queries(record, query_type):
    paragraph = record["paragraph"]
    question = record["question"]
    choices = record["choices"]
    
    if query_type == "question":
        return [question]
    if query_type == "paragraph_question":
        return [f"{paragraph} {question}"]
    if query_type == "paragraph_question_choices":
        return [" ".join([paragraph, question, *choices])]
    if query_type == "per_choice":
        return [f"{question} {choice}" for choice in choices]
    
    raise ValueError(f"query_type을 확인해주세요 (현재 query_type {query_type})")
    


def retrieve(retriever, record, query_type, k):
    """상위 k개 chunk_id 리스트 (점수 내림차순)"""
    best = {}  # chunk_id → 여러 쿼리 중 가장 높은 점수
    for query in build_queries(record, query_type):
        for chunk_id, score in retriever.search(query, k):
            best[chunk_id] = max(score, best.get(chunk_id, 0.0))
    return sorted(best, key=best.get, reverse=True)[:k]


def is_hit(record, retrieved, k, metric):
    gold_chunks = {e["chunk_id"] for e in record["evidence"]}
    gold_docs = {e["doc_id"] for e in record["evidence"]}
    top_chunks = set(retrieved[:k])
    top_docs = {chunk_id.rsplit("-", 1)[0] for chunk_id in top_chunks}
    
    if metric == "chunk_any":
        return bool(gold_chunks & top_chunks) # 교집합이 비어있지 않기만 하면 True
    if metric == "chunk_all":
        return gold_chunks <= top_chunks # 근거 청크가 모두 상위 k개 안에 포함되어있을 경우 True
    if metric == "doc_any":
        return bool(gold_docs & top_docs)
    if metric == "doc_all":
        return gold_docs <= top_docs # 근거 문서가 모두 상위 k개 안에 포함되어있을 경우 True
    raise ValueError(f"metric을 확인해주세요 (현재 metric: {metric})")


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)["oracle"]
    records = load_eval_set(config["data"]["eval_set_path"], config["split"], config.get("limit"))
    retriever = BM25Retriever()
    logger.info(f"{config['split']} {len(records)}문제")

    for query_type in QUERY_TYPES:
        retrieved = [retrieve(retriever, record, query_type, max(K_LIST)) for record in records]
        for metric in ("chunk_any", "chunk_all", "doc_any", "doc_all"):
            row = []
            for k in K_LIST:
                recall = sum(is_hit(r, ret, k, metric) for r, ret in zip(records, retrieved)) / len(records)
                row.append(f"@{k} {recall:.3f}")
            logger.info(f"{query_type:28s} {metric:9s} " + "  ".join(row))


if __name__ == "__main__":
    main()