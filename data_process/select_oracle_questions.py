"""
Oracle 평가셋 2단계: 문제 선별

- questions_all.jsonl에서 불량 문제를 제외하고 questions_selected.jsonl로 저장
- 제외 사유
    - mismatch: 질문과 선택지가 서로 다른 문제에서 온 경우 (KMMLU 원본 데이터 오류)
    - broken_format: 지문/선택지가 잘리거나 섞였거나, 원본 그림·표 없이는 풀 수 없는 경우
    - wrong_answer: 정답 레이블이 사실과 다른 경우
    - passage_only: 외부 지식 없이 지문만으로 풀리는 경우 (검색이 필요 없어 평가 목적에 맞지 않음)
- 판정 출처
    - KMMLU 90문제: 수동 검수 결과(EXCLUDED)를 그대로 사용
    - 나머지: LLM 필터(llm_filter_questions.py) 결과 사용
        - mismatch, broken_format, passage_only는 Haiku 판정으로 제외 (사전지식 의존도가 낮음)
        - wrong_answer는 Haiku와 Opus가 모두 오류라고 할 때만 제외
          (한 모델의 오판으로 LLM이 모르는 문제만 빠져 평가셋이 쉬운 쪽으로 편향되는 것을 줄이기 위함)
"""

from collections import Counter
import json
from pathlib import Path

from loguru import logger


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "oracle_eval"
INPUT_PATH = DATA_DIR / "questions_all.jsonl"
OUTPUT_PATH = DATA_DIR / "questions_selected.jsonl"
PRIMARY_RESULT_PATH = DATA_DIR / "llm_filter_results_claude-haiku-5-5.jsonl"
RECHECK_RESULT_PATH = DATA_DIR / "llm_filter_results_claude-opus-5-5.jsonl"

EXCLUDED = {
    # 질문과 선택지 불일치
    "kmmlu-korean_history-dev-0": ("mismatch", "대한광복회 질문에 한양(서울) 관련 선택지"),
    "kmmlu-korean_history-test-15": ("mismatch", "순종 재위 질문에 이승만 정부 관련 선택지"),
    "kmmlu-korean_history-test-40": ("mismatch", "순종 재위 질문에 조선 후기 농업 관련 선택지"),
    "kmmlu-korean_history-test-45": ("mismatch", "발해 질문에 조선 후기 관련 선택지"),
    "kmmlu-korean_history-test-49": ("mismatch", "향·부곡 질문에 조선 외교 관련 선택지"),
    "kmmlu-korean_history-test-57": ("mismatch", "삼국 사건 순서 질문에 조선 지도 관련 선택지"),
    "kmmlu-korean_history-test-58": ("mismatch", "임시정부 질문에 개항기 사건 선택지"),
    "kmmlu-korean_history-test-75": ("mismatch", "동학농민운동 순서 질문에 조선 후기 상업 선택지"),
    "kmmlu-korean_history-test-82": ("mismatch", "삼국 사건 순서 질문에 고려 문화재 선택지"),
    "kmmlu-korean_history-test-85": ("mismatch", "우왕 질문에 물산장려운동 선택지"),
    "kmmlu-korean_history-test-90": ("mismatch", "향·부곡 질문에 조선왕조실록 선택지"),
    "kmmlu-korean_history-test-96": ("mismatch", "김춘추 청병 질문에 여진 관련 선택지"),
    # 형식 깨짐
    "kmmlu-korean_history-dev-3": ("broken_format", "헌법 조문이 질문이 아닌 선택지 2번에 섞여 들어감"),
    "kmmlu-korean_history-test-38": ("broken_format", "연표 그림이 텍스트로 뭉개져 (가)~(라) 구간을 알 수 없음"),
    "kmmlu-korean_history-test-81": ("broken_format", "정답 선택지(4번)가 중간에 잘림"),
    # 정답 레이블 오류
    "kmmlu-korean_history-test-4": ("wrong_answer", "통감부 설치(1906)는 고종 재위 → 정답 4, 레이블 1"),
    "kmmlu-korean_history-test-43": ("wrong_answer", "직지(1377)는 우왕 대, 철령위 통보(1388) → 정답 4, 레이블 3"),
    "kmmlu-korean_history-test-55": ("wrong_answer", "고려 의학서는 향약구급방 → 정답 2, 레이블 3"),
    "kmmlu-korean_history-test-72": ("wrong_answer", "10정→2군 6위→5위→5군영 → 정답 4, 레이블 1"),
    "kmmlu-korean_history-test-95": ("wrong_answer", "탁지아문 일원화는 갑오개혁 → 정답 2, 레이블 3"),
    "kmmlu-korean_history-test-98": ("wrong_answer", "무령왕은 22담로에 왕족 파견 → 정답 3, 레이블 4"),
    "kmmlu-korean_history-test-99": ("wrong_answer", "고구려는 동맹·국동대혈 → 정답 3, 레이블 1"),
}


def load_jsonl(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f)}


def get_exclude_reason(record, primary, recheck):
    """제외 사유를 반환하고, 남길 문제면 None"""
    if record["source"] == "kmmlu":
        return EXCLUDED[record["qid"]][0] if record["qid"] in EXCLUDED else None

    judged = primary.get(record["qid"])
    if judged is None:
        return "unjudged"  # LLM 판정이 실패한 문제는 검증되지 않았으므로 제외
    for check in ("mismatch", "broken_format", "passage_only"):
        if judged[check]:
            return check
    if judged["wrong_answer"]:
        rechecked = recheck.get(record["qid"])
        if rechecked is None:
            return "unjudged"  # 재판정 전이면 제외 (llm_filter_questions.py --recheck-from 실행 필요)
        if rechecked["wrong_answer"]:
            return "wrong_answer"
    return None


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    qids = {r["qid"] for r in records}
    unknown = set(EXCLUDED) - qids
    if unknown:
        raise ValueError(f"questions_all.jsonl에 없는 qid가 제외 목록에 있음: {sorted(unknown)}")

    primary = load_jsonl(PRIMARY_RESULT_PATH)
    recheck = load_jsonl(RECHECK_RESULT_PATH)

    selected = []
    removed = Counter()
    for record in records:
        reason = get_exclude_reason(record, primary, recheck)
        if reason:
            removed[(record["source"], reason)] += 1
        else:
            selected.append(record)

    haiku_only_wrong = sum(
        1 for qid, r in recheck.items() if qid in primary and primary[qid]["wrong_answer"] and not r["wrong_answer"]
    )

    logger.info(f"전체 {len(records)}문제 → 선별 {len(selected)}문제 (제외 {len(records) - len(selected)})")
    logger.info(f"출처/제외 사유별: {dict(sorted(removed.items()))}")
    logger.info(f"Haiku만 정답 오류로 판정하고 Opus가 정상으로 본 문제(유지): {haiku_only_wrong}")
    logger.info(f"출처/과목별: {dict(sorted(Counter((r['source'], r['subject']) for r in selected).items()))}")
    logger.info(f"정답 번호 분포: {dict(sorted(Counter(r['answer'] for r in selected).items()))}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
