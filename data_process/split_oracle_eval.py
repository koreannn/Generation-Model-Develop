"""
Oracle 평가셋 마지막 단계: 문제·정답 근거 레이블을 합치고 dev/test로 나눔

- 입력: questions_selected.jsonl, gold_labels_{model}.jsonl (label_gold_evidence.py)
- 출력: data/oracle_eval/oracle_eval_set.jsonl (한 줄 = 문제 하나)
    문제 필드(qid, source, subject, paragraph, question, choices, answer, explanation)
    + split(dev|test), gold_status, relation, evidence([{doc_id, chunk_id, title, sentence}])
    - gold_status가 gold인 문제만 검색 평가(Recall@k)와 Oracle 실험에 사용
- 분할
    - dev 30% / test 70%. 청크 크기, top-k, 프롬프트 등 조정은 dev에서만 하고 test는 최종 수치에만 사용
    - (출처, 과목, 그룹) 단위로 층화. 그룹: 근거 문서 1개 gold / 근거 문서 여러 개 gold / gold 아님
    - 레이블링 파일럿 50문제는 확인 단계 프롬프트 수정에 썼으므로 dev에 고정
"""

from collections import Counter, defaultdict
import json
import random

from build_gold_candidates import DATA_DIR, QUESTIONS_PATH, load_questions
from label_gold_evidence import label_path
from loguru import logger


OUTPUT_PATH = DATA_DIR / "oracle_eval_set.jsonl"
LABEL_MODEL = "claude-haiku-5-5"
DEV_RATIO = 0.3
PILOT_SIZE = 50
SEED = 0


def group_of(label):
    if label["status"] != "gold":
        return "not_gold"
    return "gold_multi" if len({e["doc_id"] for e in label["evidence"]}) > 1 else "gold_single"


def split(records, labels, pilot_qids):
    """qid → dev|test"""
    strata = defaultdict(list)
    for record in records:
        strata[(record["source"], record["subject"], group_of(labels[record["qid"]]))].append(record["qid"])

    rng = random.Random(SEED)
    splits = {}
    for key in sorted(strata):
        qids = sorted(strata[key])
        rng.shuffle(qids)
        pilot = [q for q in qids if q in pilot_qids]
        rest = [q for q in qids if q not in pilot_qids]
        n_dev = max(round(len(qids) * DEV_RATIO) - len(pilot), 0)
        for qid in pilot + rest[:n_dev]:
            splits[qid] = "dev"
        for qid in rest[n_dev:]:
            splits[qid] = "test"
    return splits


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    with open(label_path(LABEL_MODEL), encoding="utf-8") as f:
        labels = {r["qid"]: r for r in map(json.loads, f)}
    missing = [r["qid"] for r in records if r["qid"] not in labels]
    if missing:
        raise ValueError(f"레이블이 없는 문제 {len(missing)}개: label_gold_evidence.py를 다시 실행해야 함")

    pilot_qids = {r["qid"] for r in load_questions(PILOT_SIZE)}
    splits = split(records, labels, pilot_qids)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            label = labels[record["qid"]]
            row = {
                **record,
                "split": splits[record["qid"]],
                "gold_status": label["status"],
                "relation": label["relation"],
                "evidence": label["evidence"] if label["status"] == "gold" else [],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter((splits[r["qid"]], group_of(labels[r["qid"]])) for r in records)
    for name in ("dev", "test"):
        n_gold = counts[(name, "gold_single")] + counts[(name, "gold_multi")]
        total = n_gold + counts[(name, "not_gold")]
        logger.info(
            f"{name}: 전체 {total}문제, gold {n_gold} "
            f"(근거 문서 1개 {counts[(name, 'gold_single')]}, 여러 개 {counts[(name, 'gold_multi')]})"
        )
    logger.info(f"파일럿 {len(pilot_qids)}문제는 모두 dev: {all(splits[q] == 'dev' for q in pilot_qids)}")
    logger.info(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
