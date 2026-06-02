from ast import literal_eval

from loguru import logger
import pandas as pd


def load_data(file_path):
    data = pd.read_csv(file_path)
    records = []
    for _, row in data.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
        }
        records.append(record)
    logger.debug(records[0])  # 첫 번째 레코드 출력 (디버깅용)
    return data, records


def classify_questions(records):
    social_keywords = ["ㄱ.", "㉠", "위의", "위 글" "단락", "본문", "밑줄 친", "(가)", "다음", "시기"]
    classifications = []

    for record in records:
        question = record["question"]
        paragraph = record["paragraph"]

        # 사회 영역 판단
        contains_social_keywords = any(keyword in question for keyword in social_keywords)

        # 각 선택지가 본문에 포함되어 있는지 확인
        choices_found_in_paragraph = {choice: choice in paragraph for choice in record["choices"]}

        # 정답이 포함된 선택지 찾기
        answer_index = record["answer"]
        answer_found_in_paragraph = False

        if answer_index is not None and 0 <= answer_index < len(record["choices"]):
            answer = record["choices"][answer_index]  # 정답 선택지
            answer_found_in_paragraph = choices_found_in_paragraph.get(answer, False)

        # if contains_social_keywords and not answer_found_in_paragraph:
        #     classification = '사회'
        if contains_social_keywords:
            classification = "사회"
        elif not contains_social_keywords and answer_found_in_paragraph:
            classification = "국어"
        else:
            classification = "불확실"  # 두 조건 모두 해당하지 않거나 모두 해당하는 경우

        classifications.append(
            {
                "id": record["id"],
                "classification": classification,
                "contains_social_keywords": contains_social_keywords,
                "answer_found_in_paragraph": answer_found_in_paragraph,
                "choices_found_in_paragraph": choices_found_in_paragraph,  # 선택지 포함 여부 추가
            }
        )

    return classifications


def main():
    file_path = "../data/train.csv"  # 파일 경로 설정
    data, records = load_data(file_path)

    classifications = classify_questions(records)

    # 결과를 데이터프레임으로 변환
    result_df = pd.DataFrame(classifications)

    # 결과를 CSV 파일로 저장
    output_file_path = "../data/classification_results.csv"  # 출력 파일 경로 설정
    result_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")  # CSV로 저장

    logger.debug(f"결과가 {output_file_path}에 저장되었습니다.")


if __name__ == "__main__":
    main()
