import json
import re
import time

from bs4 import BeautifulSoup
from datasets import load_dataset
from loguru import logger
import pandas as pd
import requests


def answer_symbol_to_int(symbol: str) -> int:
    # 정규표현식으로 특수문자만 추출
    special_chars = re.findall(r"[①②③④⑤]", symbol)
    cleaned_symbol = special_chars[0] if special_chars else symbol

    answer_map = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
    return answer_map.get(cleaned_symbol, -1)


def extract_question_data(soup, with_table=False):
    questions_with_table = []
    questions_without_table = []

    for item in soup.select("#examList > li"):
        # 문제 번호와 질문
        question_element = item.select_one(".pr_problem")
        question_text = question_element.get_text(strip=True)

        # 테이블 존재 여부 확인
        has_table = False

        # 문제에 테이블이 있는 경우
        question_table = question_element.find("table")
        if question_table:
            has_table = True
            question_text = {"text": question_text, "table_html": str(question_table)}

        # 문제 설명 (있는 경우)
        example = item.select_one(".exampleCon")
        example_text = example.get_text(strip=True) if example else ""

        # 예시에 테이블이 있는 경우
        if example:
            example_table = example.find("table")
            if example_table:
                has_table = True
                example_text = {"text": example_text, "table_html": str(example_table)}

        # '그림' 포함된 문제는 건너뛰기
        if isinstance(question_text, str) and "그림" in question_text:
            continue
        if isinstance(example_text, str) and "그림" in example_text:
            continue

        # 선택지를 리스트로 저장
        choices = []
        for choice in item.select(".questionCon li label"):
            choices.append(choice.get_text(strip=True))

        # 정답 번호 추출
        answer_element = item.select_one(".answer_num")
        answer_text = answer_element.get_text(strip=True).strip() if answer_element else ""
        answer = answer_symbol_to_int(answer_text)

        # 정답 설명
        explanation = item.select_one(".answer_explan")
        explanation_text = explanation.get_text(strip=True) if explanation else ""

        # 데이터 저장
        question_data = {
            "question": question_text,
            "paragraph": example_text,
            "choices": json.dumps(choices, ensure_ascii=False),
            "answer": answer,
            "answer_explanation": explanation_text,
        }

        # 테이블 유무에 따라 다른 리스트에 저장
        if has_table:
            questions_with_table.append(question_data)
        else:
            questions_without_table.append(question_data)

    return pd.DataFrame(questions_without_table) if not with_table else pd.DataFrame(questions_with_table)


def get_wr_ids(subject_code):
    # 회계학 문제
    if subject_code == 20:
        wr_ids = [903, 27, 889, 887, 885, 884, 880]

    # 헌법 문제
    if subject_code == 26:
        wr_ids = (
            [1136, 1063, 963, 953, 875, 849, 543, 537, 526, 515, 510, 500, 542, 536, 525, 514, 509]
            + [499, 541, 535, 524, 513, 508, 498, 540, 534, 523, 512, 507, 497, 539, 533, 522, 511, 506]
            + [496, 538, 532, 521, 505, 495, 531, 520, 504, 494, 530, 519, 503, 493, 529, 518, 502, 492]
            + [528, 517, 501, 491, 527, 516, 490]
        )

    # 한국사 문제
    if subject_code == 34:
        # 9급만 필터링
        wr_ids = (
            list(range(808, 814))
            + list(range(224, 243))
            + list(range(264, 269))
            + list(range(288, 298))
            + list(range(305, 326))
            + [16, 841, 870, 897, 897, 912, 962, 1012, 1026, 1053, 1167, 1172, 1173, 1176, 1177, 1184]
            + [1205, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1257, 1262, 1357, 1367, 1368, 1404, 1405]
        )

    # 사회 문제
    if subject_code == 35:
        wr_ids = list(range(808, 840)) + [865, 894, 908, 1010, 1027, 1050]

    return wr_ids


def crawl(subject_code, wr_ids=None, delay=1.0):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = "https://gichulpass.com/bbs/board.php"

    if wr_ids is None:
        wr_ids = get_wr_ids(subject_code)

    dfs = []
    for wr_id in dict.fromkeys(wr_ids):  # 중복 wr_id 제거 (순서 유지)
        params = {"bo_table": "exam", "wr_id": wr_id, "subject": subject_code}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"subject={subject_code}, wr_id={wr_id} 요청 실패: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        df = extract_question_data(soup)
        df["wr_id"] = wr_id
        dfs.append(df)
        time.sleep(delay)  # 사이트 부하 방지

    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()


def crawl_and_save(subject_code):
    concated_df = crawl(subject_code)
    len_df = len(concated_df)
    concated_df.to_csv(f"gichulpass_{subject_code}_{len_df}_raw.csv", index=False, encoding="utf-8")


def check_KMMLU(input_file, output_file):
    df = pd.read_csv(input_file, encoding="utf-8")
    ds = load_dataset("HAERAE-HUB/KMMLU", "Korean-History")
    # df의 None값을 빈 문자열로 변환하고 모든 값을 문자열로 변환
    df = df.fillna("")
    df = df.astype(str)

    # 모든 split의 question을 하나로 합치기
    questions = pd.concat([pd.DataFrame(ds["train"]), pd.DataFrame(ds["dev"]), pd.DataFrame(ds["test"])])

    # 포함 여부를 확인하는 함수
    def check_inclusion(column):
        return column.apply(lambda x: any(x in question for question in questions["question"]))

    # paragraph와 question 각각 확인 후 둘 중 하나라도 True면 True
    df["include"] = check_inclusion(df["paragraph"]) | check_inclusion(df["question"])

    new_df = df[~df["include"]]
    new_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    crawl_and_save(subject_code=20)
    crawl_and_save(subject_code=26)
    crawl_and_save(subject_code=34)
    crawl_and_save(subject_code=35)
