from data_process.process_google_translate import TranslationCache, translate_column, translate_list_column
from datasets import load_dataset
from loguru import logger
import pandas as pd
from tqdm import tqdm


def dataset_to_pd(data_name):
    data = load_dataset(data_name)
    dfs = [
        pd.DataFrame(data["murder_mysteries"]),
        pd.DataFrame(data["object_placements"]),
        pd.DataFrame(data["team_allocation"]),
    ]
    return pd.concat(dfs, axis=0)


def process_data(df):
    return pd.DataFrame(
        {
            "paragraph": df["narrative"],
            "question": df["question"],
            "choices": df["choices"],
            "answer": df["answer_index"].apply(lambda x: x + 1),
        }
    )


def process_external_datasets(dataset_name, output_filename):
    df = dataset_to_pd(dataset_name)
    df = process_data(df)

    # 개행문자를 \n 문자열로 변환
    for col in df.columns:
        if df[col].dtype == "object":  # 문자열 컬럼에 대해서만 처리
            df[col] = df[col].str.replace("\n", "\\n")

    df.to_csv(output_filename, index=False)


def translate_df(input_filename, output_filename):
    df = pd.read_csv(input_filename)

    logger.info("단락 번역 중...")
    with TranslationCache("paragraph_cache.json") as paragraph_cache:
        df["paragraph"] = translate_column(df["paragraph"], paragraph_cache)

    logger.info("질문 번역 중...")
    with TranslationCache("question_cache.json") as question_cache:
        df["question"] = translate_column(df["question"], question_cache)

    logger.info("선택지 번역 중...")
    with TranslationCache("choices_cache.json") as choices_cache:
        tqdm.pandas(desc="선택지 번역 중")
        df["choices"] = df["choices"].apply(lambda x: translate_list_column(x, choices_cache))

    df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    dataset_name = "TAUR-Lab/MuSR"
    process_external_datasets(dataset_name, "MuSR_en_raw.csv")
    translate_df("MuSR_en_raw.csv", "MuSR_ko_raw.csv")
