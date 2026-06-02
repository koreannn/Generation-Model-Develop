from datasets import load_dataset
import pandas as pd


def dataset_to_pd(data_name):
    data = load_dataset(data_name)
    return pd.DataFrame(data["test"])


def process_query_data(input_df):
    def _split_query_data(df):
        paragraphs = []
        questions = []

        for index, row in df.iterrows():
            text = row["query"]
            # Paragraph와 나머지 텍스트 분리
            paragraph, rest = text.split("Q:", 1)
            # Question과 Answer Choices 분리
            question, choices = rest.split("Answer Choices: ", 1)

            paragraphs.append(paragraph.strip())
            questions.append(question.strip())

        return pd.DataFrame(
            {
                "paragraph": paragraphs,
                "question": questions,
                "choices": df["choices"],
                "answer": df["gold"].apply(lambda x: x[0] + 1),  # gold 배열을 풀고 +1
            }
        )

    def _split_answer_choices(df):
        def process_choices(choices_list):  # 리스트 형태로 입력 받음
            new_choices = []
            for choice in choices_list:
                new_choice = (
                    choice.replace("(A)", "")
                    .replace("(B)", "")
                    .replace("(C)", "")
                    .replace("(D)", "")
                    .replace("(E)", "")
                )
                new_choices.append(new_choice.strip())
            return new_choices

        df["choices"] = df["choices"].apply(process_choices)
        return df

    split_df = _split_query_data(input_df)
    final_df = _split_answer_choices(split_df)
    return final_df


def process_and_concat_external_datasets(dataset_names, output_filename):
    dfs = []
    for dataset_name in dataset_names:
        df = dataset_to_pd(dataset_name)
        df = process_query_data(df)
        dfs.append(df)

    concated_df = pd.concat(dfs, axis=0)
    concated_df.to_csv(output_filename, index=False)


def clean_string(text):
    # 문자열 내부의 모든 큰따옴표를 작은따옴표로 변환
    text = text.replace('"', "'")
    # 연속된 큰따옴표를 하나로 변환
    while "''" in text:
        text = text.replace("''", "'")
    text = text.strip()  # 앞뒤 공백 제거
    return text


if __name__ == "__main__":
    dataset_names = [
        "dmayhem93/agieval-sat-en",
        "dmayhem93/agieval-logiqa-en",
        "dmayhem93/agieval-lsat-rc",
        "dmayhem93/agieval-lsat-lr",
        "dmayhem93/agieval-lsat-ar",
        "dmayhem93/agieval-gaokao-english",
    ]
    process_and_concat_external_datasets(dataset_names, "sat_gaokao_en_raw.csv")
