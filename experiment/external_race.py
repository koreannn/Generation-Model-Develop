from datasets import load_dataset
import pandas as pd


def dataset_to_pd(data_name):
    """주어진 데이터셋 이름으로부터 DataFrame을 생성합니다."""
    dataset = load_dataset(data_name, "high", split="validation")  # 'train', 'validation', 'test' 중 선택
    return pd.DataFrame(dataset)


def process_query_data(df):
    """DataFrame을 처리하여 필요한 형식으로 변환합니다."""

    # answer를 A, B, C, D 형식에서 1, 2, 3, 4 형식으로 변환
    def convert_answer(answer):
        if answer == "A":
            return 1
        elif answer == "B":
            return 2
        elif answer == "C":
            return 3
        elif answer == "D":
            return 4
        else:
            return None  # 예상치 못한 값에 대한 처리

    df["article"] = (
        df["article"]
        .str.replace(",", "")
        .str.replace('""', "")
        .str.replace(r"\. ", ".")
        .str.replace(r'\." ', '."')
        .str.replace(r"([.!?]) ", r"\1")
    )
    # 문제를 문자열로 변환하여 DataFrame 생성
    problems = df.apply(
        lambda row: {"question": row["question"], "choices": row["options"], "answer": convert_answer(row["answer"])},
        axis=1,
    )

    return pd.DataFrame(
        {
            "id": df["example_id"],  # 예제의 ID
            "paragraph": df["article"],  # 정리된 article 사용
            "problems": problems.apply(str),  # 문제를 문자열로 변환
            "question_plus": None,  # question_plus가 원본 데이터에 없다고 가정
        }
    )


if __name__ == "__main__":
    dataset_name = "ehovy/race"  # 사용할 데이터셋 이름
    df = dataset_to_pd(dataset_name)  # 데이터셋을 DataFrame으로 변환
    processed_df = process_query_data(df)  # 데이터 처리

    # 결과를 CSV 파일로 저장 (따옴표 처리)
    processed_df.to_csv("processed_race_dataset.csv", index=False)

    import re

    import pandas as pd

    # CSV 파일 읽기
    df = pd.read_csv("processed_race_dataset.csv")

    # paragraph 열만 수정
    df["paragraph"] = df["paragraph"].apply(lambda x: re.sub(r"\n", "", x))
    df["paragraph"] = df["paragraph"].apply(lambda x: re.sub(r"\. ", ".", x))
    df["paragraph"] = df["paragraph"].apply(lambda x: re.sub(r'\." ', '."', x))
    df["paragraph"] = df["paragraph"].apply(lambda x: re.sub(r"([.!?]) ", r"\1", x))

    # 수정된 DataFrame을 다시 CSV로 저장
    df.to_csv("modified_file.csv", index=False)
