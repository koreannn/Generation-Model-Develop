"""
정답 근거 레이블링 1단계: 문제마다 근거 후보 청크 모으기

- 입력: data/oracle_eval/questions_selected.jsonl, data/wiki/bm25_index/ (bm25_index.py)
- 출력
    - data/oracle_eval/gold_title_proposals_{model}.jsonl: LLM이 제안한 위키 문서 제목·키워드
      (qid 단위 누적, 재실행 시 이어서)
    - data/oracle_eval/gold_candidates.jsonl: 문제마다 후보 청크 목록과 각 후보를 찾은 경로
- 후보를 모으는 경로 (합집합)
    - llm_title: LLM에게 문제와 정답을 주고 근거가 있을 위키 문서 제목을 제안받아 문서를 찾음
      문서 안에서는 도입부 청크 + 문제 쿼리와 BM25 점수가 높은 청크까지 PER_DOC_CHUNKS개 (짧은 문서는 전부)
    - bm25_question: 지문 + 질문
    - bm25_choice{i}: 질문 + i번 선택지 (개체 이름이 선택지에만 있는 문제를 위함)
    - bm25_keywords: LLM이 뽑은 키워드
  BM25로만 후보를 모으면 BM25가 찾을 수 있는 문서만 정답이 되어 이후 검색기 비교가 BM25 쪽으로 기울어지므로,
  검색기와 무관한 LLM 제목 제안을 함께 씀
- 후보가 MAX_CANDIDATES개를 넘으면 두 경로(LLM, BM25)가 모두 찾은 청크 → 각 경로 안의 순위 순으로 남김

사용법
    # 파일럿: 무작위 50문제
    uv run python data_process/build_gold_candidates.py --pilot 50
    # 전체
    uv run python data_process/build_gold_candidates.py
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import anthropic
from bm25_index import CHUNKS_PATH, BM25Index
from build_wiki_redirects import REDIRECTS_PATH
from dotenv import load_dotenv
from llm_filter_questions import PRICES, format_question, report_cost, request_json, run_parallel
from loguru import logger


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "oracle_eval"
QUESTIONS_PATH = DATA_DIR / "questions_selected.jsonl"
CANDIDATES_PATH = DATA_DIR / "gold_candidates.jsonl"

MAX_TITLES = 5
PER_DOC_CHUNKS = 4
QUESTION_K = 15
CHOICE_K = 5
KEYWORDS_K = 10
MAX_CANDIDATES = 30
PILOT_SEED = 0

PROPOSAL_PROMPT = """\
당신은 한국어 위키백과에 익숙한 검색 보조자입니다.
객관식 문제와 정답 번호가 주어집니다. 정답이 맞다는 것을 확인할 수 있는 근거
(옳지 않은 것을 고르는 문제라면 정답 선택지가 틀렸다는 근거)가 실려 있을 한국어 위키백과 문서를 찾으려고 합니다.

- titles: 근거가 있을 가능성이 높은 한국어 위키백과 문서 제목을 가능성이 높은 순서로 1~5개.
  실제 표제어 표기를 따르고, 같은 이름의 문서가 여럿이면 괄호 구분까지 적으세요.
- keywords: 근거 문장에 나올 법한 핵심 개체명·용어 3~8개.\
"""

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["titles", "keywords"],
    "additionalProperties": False,
}


def load_questions(pilot=None):
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if pilot:
        records = random.Random(PILOT_SEED).sample(records, pilot)
    return records


def proposal_path(model):
    return DATA_DIR / f"gold_title_proposals_{model}.jsonl"


def propose(client, model, record):
    result = request_json(client, model, PROPOSAL_PROMPT, format_question(record), PROPOSAL_SCHEMA, effort="low")
    result["qid"] = record["qid"]
    result["titles"] = result["titles"][:MAX_TITLES]
    return result


def normalize_title(title):
    return re.sub(r"\s+", "", title).lower()


def load_titles():
    """(정규화한 제목 → doc_id, 정규화한 별칭 → doc_id)

    제목은 청크 파일의 문서 첫 청크 기준, 별칭은 리다이렉트 중 대상 문서가 코퍼스에 있는 것만
    """
    titles = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["order"] == 0:
                titles.setdefault(normalize_title(chunk["title"]), chunk["doc_id"])
    with open(REDIRECTS_PATH, encoding="utf-8") as f:
        redirects = json.load(f)
    aliases = {}
    for alias, target in redirects.items():
        if normalize_title(target) in titles:
            aliases[normalize_title(alias)] = titles[normalize_title(target)]
    return titles, aliases


def resolve_title(title, titles, aliases):
    """(doc_id 또는 None, 찾은 방식)"""
    # "중양절 (명절)"처럼 LLM이 붙인 괄호 구분을 떼면 일치하는 경우도 확인
    keys = [normalize_title(title), normalize_title(re.sub(r"\s*\(.*?\)\s*$", "", title))]
    for key in keys:
        if key in titles:
            return titles[key], "exact"
    for key in keys:
        if key in aliases:
            return aliases[key], "redirect"
    return None, "unresolved"


def collect(record, proposal, index, titles, aliases):
    question = record["question"]
    query_all = " ".join([question, *record["choices"], *proposal["keywords"]])
    ranked_lists = {}  # 경로 → [chunk_id, ...] (경로 안 순위 순)

    scores_all = index.scores(query_all)
    matched = []
    llm_chunks = []
    for title in proposal["titles"]:
        doc_id, match = resolve_title(title, titles, aliases)
        matched.append({"title": title, "doc_id": doc_id, "match": match})
        if doc_id is None:
            continue
        rows = index.doc_rows[doc_id]
        lead = index.chunk_ids[rows[0]]
        ranked = [chunk_id for chunk_id, _ in index.rank_in_doc(scores_all, doc_id, len(rows))]
        llm_chunks += [lead] + [c for c in ranked if c != lead][: PER_DOC_CHUNKS - 1]
    ranked_lists["llm_title"] = list(dict.fromkeys(llm_chunks))

    ranked_lists["bm25_question"] = [c for c, _ in index.search(f"{record['paragraph']} {question}", QUESTION_K)]
    for i, choice in enumerate(record["choices"], 1):
        ranked_lists[f"bm25_choice{i}"] = [c for c, _ in index.search(f"{question} {choice}", CHOICE_K)]
    if proposal["keywords"]:
        ranked_lists["bm25_keywords"] = [c for c, _ in index.search(" ".join(proposal["keywords"]), KEYWORDS_K)]

    pooled = {}
    for path, chunk_ids in ranked_lists.items():
        for rank, chunk_id in enumerate(chunk_ids):
            entry = pooled.setdefault(chunk_id, {"chunk_id": chunk_id, "paths": [], "best_rank": rank})
            entry["paths"].append(path)
            entry["best_rank"] = min(entry["best_rank"], rank)

    def priority(entry):
        sources = {"llm" if p == "llm_title" else "bm25" for p in entry["paths"]}
        return (-len(sources), entry["best_rank"])

    candidates = sorted(pooled.values(), key=priority)[:MAX_CANDIDATES]
    for entry in candidates:
        del entry["best_rank"]
    return {
        "qid": record["qid"],
        "titles": matched,
        "keywords": proposal["keywords"],
        "n_pooled": len(pooled),
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, help="무작위 N문제만 처리 (고정 시드)")
    parser.add_argument("--model", choices=sorted(PRICES), default="claude-haiku-5-5")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    load_dotenv(ROOT / "config" / ".env")  # ANTHROPIC_API_KEY
    records = load_questions(args.pilot)

    client = anthropic.Anthropic()
    proposals = run_parallel(lambda r: propose(client, args.model, r), records, proposal_path(args.model), args.workers)
    report_cost(args.model, [proposals[r["qid"]] for r in records if r["qid"] in proposals], len(records))

    logger.info("BM25 인덱스와 문서 제목 로드")
    index = BM25Index()
    titles, aliases = load_titles()

    results = []
    for i, record in enumerate(records, 1):
        if record["qid"] not in proposals:
            continue
        results.append(collect(record, proposals[record["qid"]], index, titles, aliases))
        if i % 200 == 0:
            logger.info(f"후보 수집 {i}/{len(records)}")

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    matches = Counter(t["match"] for r in results for t in r["titles"])
    paths = Counter(p for r in results for c in r["candidates"] for p in c["paths"])
    logger.info(f"제안 제목 {sum(matches.values())}개 매칭: {dict(matches)}")
    logger.info(
        f"문제당 후보 평균 {sum(len(r['candidates']) for r in results) / len(results):.1f}개 (합치기 전 평균 "
        f"{sum(r['n_pooled'] for r in results) / len(results):.1f}개)"
    )
    logger.info(f"후보 경로별 개수: {dict(sorted(paths.items()))}")
    logger.info(f"저장 완료: {CANDIDATES_PATH} ({len(results)}문제)")


if __name__ == "__main__":
    main()
