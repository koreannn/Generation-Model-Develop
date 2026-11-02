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
    """쿼리 문자열 리스트. per_choice만 여러 개, 나머지는 1개"""
    # TODO
    # - record["paragraph"], record["question"], record["choices"]를 공백으로 이어 붙이기
    # - paragraph가 빈 문자열인 문제도 있으니 그대로 이어 붙여도 됨 (BM25는 공백을 무시)
    # - per_choice: [f"{question} {choice}" for choice in choices]
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
    """상위 k개(retrieved[:k])가 metric 기준으로 맞았는지"""
    # TODO
    # - 근거 청크 집합: {e["chunk_id"] for e in record["evidence"]}
    # - 근거 문서 집합: {e["doc_id"] for e in record["evidence"]}
    # - 검색된 청크의 doc_id: chunk_id.rsplit("-", 1)[0]
    # - chunk_any: 교집합이 있으면 True / chunk_all: 근거 청크가 전부 포함되면 True / doc_any: 문서 교집합이 있으면 True
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
    raise ValueError(f"metric을 확인해주세요 (현재 metric: {metric})")


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)["oracle"]
    records = load_eval_set(config["data"]["eval_set_path"], config["split"], config.get("limit"))
    retriever = BM25Retriever()
    logger.info(f"{config['split']} {len(records)}문제")

    for query_type in QUERY_TYPES:
        retrieved = [retrieve(retriever, record, query_type, max(K_LIST)) for record in records]
        for metric in ("chunk_any", "chunk_all", "doc_any"):
            row = []
            for k in K_LIST:
                recall = sum(is_hit(r, ret, k, metric) for r, ret in zip(records, retrieved)) / len(records)
                row.append(f"@{k} {recall:.3f}")
            logger.info(f"{query_type:28s} {metric:9s} " + "  ".join(row))


if __name__ == "__main__":
    main()