"""
정답 근거 레이블링 2단계: 후보 청크에서 정답 근거 문장 고르기

- 입력: data/oracle_eval/gold_candidates.jsonl (build_gold_candidates.py 출력)
- 출력
    - data/oracle_eval/gold_labels_{model}.jsonl: 문제마다 레이블 (qid 단위 누적, 재실행 시 이어서)
    - data/oracle_eval/gold_review_{model}.md: 사람 검수용 문서 (이번에 처리한 문제)
- 문제마다 LLM 호출 두 번
    1. 레이블링: 문제, 정답 번호, 후보 청크를 주고 정답을 확정하는 문장을 원문 그대로 고르게 함
       - supports: 정답 선택지가 맞다는 근거 / refutes: 옳지 않은 것을 고르는 문제에서 정답 선택지가 틀렸다는 근거
       - 후보 안에 근거가 없으면 none, 확신이 없으면 uncertain
    2. 근거가 가리키는 선택지 확인: 정답을 알려주지 않고 문제와 근거 문장만 주어,
       근거가 뒷받침(supports)하거나 반박(refutes)하는 선택지를 고르게 함
       정답을 알려주고 근거를 찾게 하면 관련만 있는 문장을 근거로 끼워 맞출 수 있어,
       근거가 정답 선택지를 특정하는지 확인
       (근거만으로 문제 전체를 풀게 하면 옳지 않은 것을 고르는 문제는 나머지 선택지의 근거가 없어 항상 실패함)
- 최종 status
    - gold: 근거 문장이 후보 청크 원문에 그대로 있고, 근거가 가리키는 선택지가 정답과 같음
    - none: 후보 안에 근거 없음 (검색 평가에서 제외하고 코퍼스 커버리지로 따로 보고)
    - uncertain: LLM이 확신하지 못함
    - rejected_verbatim: 근거 문장이 원문에 없음 (LLM이 문장을 바꾸거나 지어냄)
    - rejected_blind: 근거가 가리키는 선택지가 정답과 다름

사용법
    uv run python data_process/label_gold_evidence.py --model claude-haiku-5-5
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import anthropic
from bm25_index import CHUNKS_PATH
from build_gold_candidates import CANDIDATES_PATH, DATA_DIR, QUESTIONS_PATH
from dotenv import load_dotenv
from llm_filter_questions import PRICES, format_question, report_cost, request_json, run_parallel
from loguru import logger


ROOT = Path(__file__).resolve().parent.parent

LABEL_PROMPT = """\
당신은 RAG 평가셋의 정답 근거를 레이블링하는 검수자입니다.
객관식 문제, 정답 번호, 번호가 붙은 한국어 위키백과 후보 청크가 주어집니다.
후보 청크에서 정답을 확정하는 근거 문장을 찾으세요.

- relation
  - supports: 옳은 것·해당하는 것을 고르는 문제. 정답 선택지가 맞다는 것을 보여주는 문장을 찾습니다.
  - refutes: 옳지 않은 것·해당하지 않는 것을 고르는 문제. 정답 선택지가 틀렸다는 것을 보여주는 문장을 찾습니다.
- evidence: 근거 문장마다 후보 번호(candidate)와 문장(sentence)을 적습니다.
  1. 문장은 후보 청크에 있는 그대로 한 글자도 바꾸지 말고 복사하세요. 각 청크의 첫 줄(제목 > 섹션)은 문장이 아닙니다.
  2. 문제와 관련만 있는 문장은 근거가 아닙니다. 그 문장을 읽고 다른 선택지가 아닌 정답 선택지를 고를 수 있어야 합니다.
     지문의 단서로 대상을 특정한 뒤 대상에 대한 사실로 정답을 확정하는 문제라면, 그 사실을 담은 문장이 근거입니다.
  3. 정답 확정에 필요한 최소한의 문장만 고르세요. 근거가 여러 청크에 흩어져 있으면 모두 적으세요.
- status
  - found: 정답을 확정하는 근거를 찾음
  - none: 후보 안에 정답을 확정하는 문장이 없음. 억지로 찾지 마세요. none도 정상적인 답입니다. evidence는 비웁니다.
  - uncertain: 근거로 보이지만 확신할 수 없음
- reason: 판단 근거를 한두 문장으로 적습니다.\
"""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["found", "none", "uncertain"]},
        "relation": {"type": "string", "enum": ["supports", "refutes"]},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"candidate": {"type": "integer"}, "sentence": {"type": "string"}},
                "required": ["candidate", "sentence"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["status", "relation", "evidence", "reason"],
    "additionalProperties": False,
}

BLIND_PROMPTS = {
    "supports": """\
객관식 문제와 근거 문장이 주어집니다. 배경지식 없이 문제(지문 포함)와 근거 문장만 읽은 사람이
정답으로 고르게 될 선택지 번호를 적으세요. 근거 문장이 어느 한 선택지를 가리키지 않으면 answer를 null로 두세요.\
""",
    "refutes": """\
객관식 문제와 근거 문장이 주어집니다. 옳지 않은 것을 고르는 문제입니다.
배경지식 없이 문제(지문 포함)와 근거 문장만 읽은 사람이 근거 문장 때문에 틀렸다고 판단할 선택지 번호를 적으세요.
근거 문장이 어느 한 선택지를 반박하지 않으면 answer를 null로 두세요.\
""",
}

BLIND_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": ["integer", "null"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


def label_path(model):
    return DATA_DIR / f"gold_labels_{model}.jsonl"


def normalize_space(text):
    return re.sub(r"\s+", " ", text).strip()


def format_without_answer(record):
    return format_question(record).rsplit("\n\n[표시된 정답]", 1)[0]


def format_candidates(candidates, chunks):
    return "\n\n".join(f"[{i}] {chunks[c['chunk_id']]['text']}" for i, c in enumerate(candidates, 1))


def label(client, model, record, candidates, chunks):
    content = f"{format_question(record)}\n\n[후보 청크]\n{format_candidates(candidates, chunks)}"
    result = request_json(client, model, LABEL_PROMPT, content, LABEL_SCHEMA)
    tokens = {"input_tokens": result.pop("input_tokens"), "output_tokens": result.pop("output_tokens")}
    out = {"qid": record["qid"], "llm_status": result["status"]}
    out.update({k: result[k] for k in ("relation", "reason")})

    evidence, mismatched = [], []
    for item in result["evidence"]:
        if not 1 <= item["candidate"] <= len(candidates):
            mismatched.append(item["sentence"])
            continue
        chunk = chunks[candidates[item["candidate"] - 1]["chunk_id"]]
        body = chunk["text"].split("\n", 1)[1]  # 헤더 줄 제외
        if normalize_space(item["sentence"]) and normalize_space(item["sentence"]) in normalize_space(body):
            evidence.append({**{k: chunk[k] for k in ("doc_id", "chunk_id", "title")}, "sentence": item["sentence"]})
        else:
            mismatched.append(item["sentence"])
    out["evidence"] = evidence
    out["mismatched"] = mismatched

    if result["status"] != "found":
        out["status"] = result["status"]
    elif mismatched or not evidence:
        out["status"] = "rejected_verbatim"
    else:
        sentences = "\n".join(f"- {e['sentence']}" for e in evidence)
        content = f"{format_without_answer(record)}\n\n[근거 문장]\n{sentences}"
        blind = request_json(client, model, BLIND_PROMPTS[result["relation"]], content, BLIND_SCHEMA)
        tokens["input_tokens"] += blind["input_tokens"]
        tokens["output_tokens"] += blind["output_tokens"]
        out["blind_answer"] = blind["answer"]
        out["status"] = "gold" if blind["answer"] == record["answer"] else "rejected_blind"
    return {**out, **tokens}


def load_chunks(chunk_ids):
    chunks = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            # 필요한 청크만 파싱 (chunk_id가 줄 맨 앞에 있음)
            chunk_id = line[len('{"chunk_id": "') : line.index('"', len('{"chunk_id": "'))]
            if chunk_id in chunk_ids:
                chunks[chunk_id] = json.loads(line)
    return chunks


def write_review(path, records, candidates, chunks, labels):
    lines = ["# 정답 근거 레이블 검수", "", "각 문제마다 판정을 적어 주세요.", ""]
    lines += ["- 정밀도: status가 gold인 문제의 근거 문장만 읽고 정답을 확정할 수 있는가", ""]
    lines += ["- none 정확도: status가 none인 문제의 후보 청크에 정말 근거가 없는가", ""]
    for n, record in enumerate(records, 1):
        result = labels.get(record["qid"])
        if result is None:
            continue
        lines += [f"## {n}. {record['qid']} — {result['status']}", "", "```", format_question(record), "```", ""]
        lines += [f"- relation: {result['relation']}, LLM 판단 근거: {result['reason']}"]
        if "blind_answer" in result:
            lines.append(f"- 근거가 가리키는 선택지: {result['blind_answer']}")
        for e in result["evidence"]:
            lines.append(f"- 근거 [{e['title']} / {e['chunk_id']}] {e['sentence']}")
        for sentence in result["mismatched"]:
            lines.append(f"- 원문에 없는 문장: {sentence}")
        lines += ["", "<details><summary>후보 청크</summary>", ""]
        for i, c in enumerate(candidates[record["qid"]], 1):
            text = chunks[c["chunk_id"]]["text"].replace("\n", " ⏎ ")
            lines.append(f"{i}. ({', '.join(c['paths'])}) {text}")
        lines += ["", "</details>", "", "**판정:** ", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(PRICES), default="claude-haiku-5-5")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    load_dotenv(ROOT / "config" / ".env")  # ANTHROPIC_API_KEY
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        candidates = {r["qid"]: r["candidates"] for r in map(json.loads, f)}
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        records = [r for r in map(json.loads, f) if r["qid"] in candidates]

    logger.info(f"{len(records)}문제의 후보 청크 로드")
    chunks = load_chunks({c["chunk_id"] for cands in candidates.values() for c in cands})

    client = anthropic.Anthropic()
    labels = run_parallel(
        lambda r: label(client, args.model, r, candidates[r["qid"]], chunks),
        records,
        label_path(args.model),
        args.workers,
    )
    done = [labels[r["qid"]] for r in records if r["qid"] in labels]
    report_cost(args.model, done, len(records))
    logger.info(f"status 분포: {dict(Counter(r['status'] for r in done))}")
    logger.info(f"relation 분포: {dict(Counter(r['relation'] for r in done if r['status'] == 'gold'))}")

    review_path = DATA_DIR / f"gold_review_{args.model}.md"
    write_review(review_path, records, candidates, chunks, labels)
    logger.info(f"검수 문서 저장: {review_path}")


if __name__ == "__main__":
    main()
