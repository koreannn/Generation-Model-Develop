"""
Oracle 평가셋 1단계: 문제 수집 및 형식 통일

- 출처별 로더가 공통 스키마의 레코드 리스트를 반환하고, 정제/중복 제거 후 jsonl로 저장
- 공통 스키마: qid, source, subject, paragraph, question, choices(list), answer(int, 1-based), explanation
- 현재 출처: KMMLU 한국사, 대회 train.csv 지식형(4지선다), CLIcK 문화 영역 (추후 공무원 기출 PDF 로더 추가 예정)
"""

import ast
from collections import Counter
import json
from pathlib import Path
import re

from datasets import load_dataset
from loguru import logger
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV_PATH = ROOT / "data" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "oracle_eval" / "questions_all.jsonl"

# CLIcK 문화 영역 문제는 "다음은 ~에 대한 문제이다." 접두어로 과목을 표시함 (언어 영역에는 없음)
CLICK_SUBJECT_PREFIXES = {
    "다음은 한국 사회의 경제에 대한 문제이다.": "economy",
    "다음은 한국의 지리에 대한 문제이다.": "geography",
    "다음은 한국의 역사에 대한 문제이다.": "korean_history",
    "다음은 한국사회의 법에 대한 문제이다.": "law",
    "다음은 한국사회의 정치에 대한 문제이다.": "politics",
    "다음은 한국의 대중문화에 대한 문제이다.": "popular_culture",
    "다음은 한국사회에 대한 문제이다.": "society",
    "다음은 한국의 문화와 전통에 대한 문제이다.": "tradition",
}


def load_kmmlu_korean_history():
    ds = load_dataset("HAERAE-HUB/KMMLU", "Korean-History")

    records = []
    for split in ("train", "dev", "test"):
        for idx, row in enumerate(ds[split]):
            records.append(
                {
                    "qid": f"kmmlu-korean_history-{split}-{idx}",
                    "source": "kmmlu",
                    "subject": "korean_history",
                    "paragraph": "",  # KMMLU는 지문이 question 안에 합쳐져 있음
                    "question": row["question"].strip(),
                    "choices": [row[c].strip() for c in ("A", "B", "C", "D")],
                    "answer": int(row["answer"]),
                    "explanation": "",
                }
            )
    return records


def load_train_csv_knowledge():
    """대회 train.csv 중 지식형(4지선다, KMMLU·MMMLU 출신)만 사용. 5지선다는 KLUE MRC 뉴스 독해형이라 제외"""
    df = pd.read_csv(TRAIN_CSV_PATH).fillna("")

    records = []
    for _, row in df.iterrows():
        problems = ast.literal_eval(row["problems"])
        if len(problems["choices"]) != 4:
            continue

        question = problems["question"].strip()
        if row["question_plus"]:
            question += "\n<보기>\n" + row["question_plus"].strip()

        records.append(
            {
                "qid": f"train_csv-{row['id']}",
                "source": "train_csv",
                "subject": "unknown",  # train.csv에는 과목 정보가 없음
                "paragraph": row["paragraph"].strip(),
                "question": question,
                "choices": [c.strip() for c in problems["choices"]],
                "answer": int(problems["answer"]),
                "explanation": "",
            }
        )
    return records


def load_click_culture():
    """CLIcK 중 문화 영역(역사, 지리, 법, 정치, 사회, 경제, 전통, 대중문화)만 사용. 언어 영역(국어, TOPIK 등)은 제외"""
    ds = load_dataset("EunsuKim/CLIcK")["train"]

    records = []
    for row in ds:
        question = row["question"].strip()
        prefix = next((p for p in CLICK_SUBJECT_PREFIXES if question.startswith(p)), None)
        if prefix is None:
            continue

        records.append(
            {
                "qid": f"click-{row['id']}",
                "source": "click",
                "subject": CLICK_SUBJECT_PREFIXES[prefix],
                "paragraph": row["paragraph"].strip(),
                "question": question[len(prefix) :].strip(),
                "choices": [c.strip() for c in row["choices"]],
                "answer": row["choices"].index(row["answer"]) + 1,  # CLIcK 정답은 선택지 텍스트
                "explanation": "",
            }
        )
    return records


def get_invalid_reason(record):
    if not record["question"]:
        return "empty_question"
    if len(record["choices"]) not in (4, 5) or any(not c for c in record["choices"]):
        return "invalid_choices"
    if not 1 <= record["answer"] <= len(record["choices"]):
        return "invalid_answer"
    return None


def normalize(text):
    return re.sub(r"\s+", "", text)


def is_duplicate(a, b):
    """
    선택지가 같고, 한쪽의 질문이 다른 쪽의 지문+질문 안에 포함되면 중복으로 판단
    - KMMLU는 지문이 질문에 합쳐져 있고 train.csv는 지문/질문이 분리되어 있어 단순 문자열 비교로는 못 잡음
    """
    if [normalize(c) for c in a["choices"]] != [normalize(c) for c in b["choices"]]:
        return False
    text_a = normalize(a["paragraph"] + a["question"])
    text_b = normalize(b["paragraph"] + b["question"])
    return normalize(a["question"]) in text_b or normalize(b["question"]) in text_a


def clean_and_dedup(records):
    """먼저 들어온 레코드를 유지 (KMMLU를 먼저 넣어 select_oracle_questions.py의 수동 검수 qid가 유지되도록 함)"""
    removed = Counter()
    kept = []
    by_choices = {}  # 선택지 기준으로 후보를 묶어 비교 횟수를 줄임

    for record in records:
        reason = get_invalid_reason(record)
        if reason:
            removed[reason] += 1
            continue

        key = normalize("".join(record["choices"]))
        if any(is_duplicate(record, other) for other in by_choices.get(key, [])):
            removed[f"duplicate_{record['source']}"] += 1
            continue

        by_choices.setdefault(key, []).append(record)
        kept.append(record)

    return kept, removed


def summarize(records, removed):
    logger.info(f"최종 문제 수: {len(records)}")
    logger.info(f"출처/과목별: {dict(Counter((r['source'], r['subject']) for r in records))}")
    logger.info(f"제거 사유별: {dict(removed)}")
    logger.info(f"정답 번호 분포: {dict(sorted(Counter(r['answer'] for r in records).items()))}")


def main():
    records = []
    for loader in (load_kmmlu_korean_history, load_train_csv_knowledge, load_click_culture):
        loaded = loader()
        logger.info(f"{loader.__name__}: {len(loaded)}문제")
        records.extend(loaded)
    logger.info(f"수집한 문제 수: {len(records)}")

    records, removed = clean_and_dedup(records)
    summarize(records, removed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
