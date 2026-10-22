"""
Oracle 평가셋 2단계 (자동화): LLM으로 불량 문제 걸러내기

- 문제마다 Claude에게 4가지를 판정하게 함
    - mismatch: 질문과 선택지가 서로 다른 문제에서 온 경우
    - broken_format: 지문/선택지가 잘리거나 섞였거나, 원본 그림·표 없이는 풀 수 없는 경우
    - wrong_answer: 정답 레이블이 사실과 다른 경우
    - passage_only: 외부 지식 없이 지문·질문 안의 정보만으로 정답을 확정할 수 있는 경우 (검색이 필요 없는 문제)
- 결과는 모델별로 llm_filter_results_{model}.jsonl에 qid 단위로 누적 저장 (중단 후 재실행하면 이어서 진행)
- 문제마다 입력/출력 토큰을 기록해 실제 비용을 계산

사용법
    # 수동 검수한 KMMLU 90문제로 필터 성능 검증
    uv run python data_process/llm_filter_questions.py --validate --model claude-haiku-5-5
    # 전체 문제에 적용
    uv run python data_process/llm_filter_questions.py --model claude-haiku-5-5
    # Haiku가 정답 오류로 판정한 문제만 Opus로 재판정 (두 모델이 모두 동의할 때만 제외하기 위함)
    uv run python data_process/llm_filter_questions.py --model claude-opus-5-5 --recheck-from claude-haiku-5-5
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading

import anthropic
from dotenv import load_dotenv
from loguru import logger
from select_oracle_questions import EXCLUDED


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "oracle_eval"
INPUT_PATH = DATA_DIR / "questions_all.jsonl"

# 100만 토큰당 가격 (입력, 출력) USD
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# 서버 측 거절 fallback을 지원하는 모델 (Haiku는 미지원)
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
CHECKS = ("mismatch", "broken_format", "wrong_answer", "passage_only")

SYSTEM_PROMPT = """\
당신은 한국어 객관식 시험 문제의 품질을 검수하는 전문가입니다.
이 문제들은 RAG(검색 증강 생성) 평가셋에 쓰일 예정이며, 외부 지식이 필요한 정상 문제만 남기려고 합니다.
문제를 보고 아래 네 가지를 판정하세요.

1. mismatch: 질문(지문)과 선택지가 서로 다른 문제에서 온 것처럼 주제가 맞지 않는가?
   예) 독립운동 단체를 묻는데 선택지가 모두 한양 도성에 대한 설명인 경우
2. broken_format: 지문이나 선택지가 중간에 잘렸거나, 서로 섞였거나, 원본에 있던 그림·표·연표 없이는 풀 수 없는가?
   단순 띄어쓰기 오류, 오탈자, 문장 끝 일부 누락처럼 풀이에 지장이 없는 문제는 해당하지 않습니다.
3. wrong_answer: 표시된 정답 번호가 사실과 다른가? 직접 풀어 보고, 확실할 때만 true로 판정하세요.
   mismatch나 broken_format 때문에 정답을 판단할 수 없으면 false로 두세요.
4. passage_only: 배경지식 없이 지문과 질문 안의 정보만 읽고 정답을 확정할 수 있는가?
   지문의 단서로 대상을 추론한 뒤 그 대상에 대한 지식이 필요하면 false입니다.

your_answer에는 직접 푼 정답 번호를 적고, 판단할 수 없으면 null로 두세요.
reason에는 true로 판정한 항목이 있을 때 그 근거를 한두 문장으로 적고, 없으면 빈 문자열로 두세요.\
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        **{check: {"type": "boolean"} for check in CHECKS},
        "your_answer": {"type": ["integer", "null"]},
        "reason": {"type": "string"},
    },
    "required": [*CHECKS, "your_answer", "reason"],
    "additionalProperties": False,
}


def format_question(record):
    lines = []
    if record["paragraph"]:
        lines += ["[지문]", record["paragraph"], ""]
    lines += ["[질문]", record["question"], "", "[선택지]"]
    lines += [f"{i}. {choice}" for i, choice in enumerate(record["choices"], 1)]
    lines += ["", f"[표시된 정답] {record['answer']}번"]
    return "\n".join(lines)


def result_path(model):
    return DATA_DIR / f"llm_filter_results_{model}.jsonl"


def judge(client, model, record):
    kwargs = {}
    if model in FALLBACK_MODELS:
        # 안전 분류기가 거절하면 서버에서 다른 모델로 재시도
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": format_question(record)}],
        **kwargs,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"모델이 판정을 거절함: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("max_tokens에 도달해 응답이 잘림")

    text = next(block.text for block in response.content if block.type == "text")
    result = json.loads(text)
    result["qid"] = record["qid"]
    result["exclude"] = any(result[check] for check in CHECKS)
    result["input_tokens"] = response.usage.input_tokens
    result["output_tokens"] = response.usage.output_tokens  # thinking 토큰 포함
    return result


def load_results(model):
    path = result_path(model)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f)}


def report_cost(model, results, n_total):
    measured = [r for r in results if "input_tokens" in r]
    if not measured:
        return
    input_price, output_price = PRICES[model]
    cost = sum(r["input_tokens"] * input_price + r["output_tokens"] * output_price for r in measured) / 1e6
    per_question = cost / len(measured)
    logger.info(
        f"[{model}] 측정 {len(measured)}문제 비용 ${cost:.3f} (문제당 ${per_question:.5f}), "
        f"{n_total}문제 전체 추정 ${per_question * n_total:.2f} (Batch API 사용 시 ${per_question * n_total / 2:.2f})"
    )


def run(model, records, workers):
    done = load_results(model)
    todo = [r for r in records if r["qid"] not in done]
    logger.info(f"대상 {len(records)}문제 중 이미 판정 {len(records) - len(todo)}, 남은 {len(todo)}")

    client = anthropic.Anthropic()
    lock = threading.Lock()
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool, open(result_path(model), "a", encoding="utf-8") as f:
        futures = {pool.submit(judge, client, model, r): r["qid"] for r in todo}
        for i, future in enumerate(as_completed(futures), 1):
            qid = futures[future]
            try:
                result = future.result()
            except (anthropic.APIError, RuntimeError, json.JSONDecodeError) as e:
                failed += 1
                logger.warning(f"{qid} 판정 실패: {e}")
                continue
            with lock:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done[qid] = result
            if i % 50 == 0:
                logger.info(f"진행 {i}/{len(todo)}")

    if failed:
        logger.warning(f"실패 {failed}문제: 다시 실행하면 실패한 문제만 재시도함")
    return done


def validate(records, results):
    """수동 검수 결과(select_oracle_questions.EXCLUDED)를 정답으로 보고 LLM 필터의 제외 판정을 비교"""
    targets = [r for r in records if r["source"] == "kmmlu" and r["qid"] in results]
    tp = fp = fn = tn = 0
    for record in targets:
        actual = record["qid"] in EXCLUDED
        predicted = results[record["qid"]]["exclude"]
        tp += actual and predicted
        fp += not actual and predicted
        fn += actual and not predicted
        tn += not actual and not predicted

        if actual != predicted:
            manual = EXCLUDED.get(record["qid"], ("정상", ""))[0]
            flags = [c for c in CHECKS if results[record["qid"]][c]] or ["정상"]
            logger.info(f"불일치 {record['qid']}: 수동={manual}, LLM={flags}, 근거={results[record['qid']]['reason']}")

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    logger.info(f"검증 {len(targets)}문제: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    logger.info(f"불량 검출 정밀도 {precision:.2f}, 재현율 {recall:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="수동 검수한 KMMLU 문제만 판정하고 성능을 출력")
    parser.add_argument("--model", choices=sorted(PRICES), default="claude-haiku-5-5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--recheck-from",
        choices=sorted(PRICES),
        help="해당 모델이 정답 오류(wrong_answer)로 판정한 문제만 다시 판정",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / "config" / ".env")  # ANTHROPIC_API_KEY
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    n_total = len(records)

    if args.validate:
        records = [r for r in records if r["source"] == "kmmlu"]
    if args.recheck_from:
        flagged = {qid for qid, r in load_results(args.recheck_from).items() if r["wrong_answer"]}
        records = [r for r in records if r["qid"] in flagged]
        logger.info(f"{args.recheck_from}가 정답 오류로 판정한 {len(records)}문제를 {args.model}로 재판정")

    results = run(args.model, records, args.workers)
    report_cost(args.model, [results[r["qid"]] for r in records if r["qid"] in results], n_total)

    if args.validate:
        validate(records, results)
    else:
        excluded = [r for r in records if results.get(r["qid"], {}).get("exclude")]
        logger.info(f"전체 {len(records)}문제 중 LLM이 제외 판정한 문제 {len(excluded)}")
        for check in CHECKS:
            logger.info(f"  {check}: {sum(1 for r in results.values() if r[check])}")


if __name__ == "__main__":
    main()
