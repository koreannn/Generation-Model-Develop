from ast import literal_eval

import dspy
import pandas as pd
from tqdm import tqdm


def process_csv(file_path, output_path, lm_api_key):
    lm = dspy.LM("openai/gpt-4o", api_key=lm_api_key)
    dspy.configure(lm=lm)

    df = pd.read_csv(file_path)

    records = []
    for _, row in df.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
            "question_plus": problems.get("question_plus", None),
        }
        records.append(record)

    df = pd.DataFrame(records)

    df["steps"] = None
    data_list = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        input_data = {
            "paragraph": row["paragraph"],
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        }
        classify = dspy.ChainOfThought("paragraph: str, question: str, choices: list, answer: str -> steps: list", n=1)

        input_data["question"] = f"{row['question']} 단계별 설명(CoT)을 사용하여 올바른 답을 도출하세요."
        response = classify(**input_data)
        print("response.completions", response.completions)
        data_list.append(response.completions)

    df["steps"] = data_list

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Updated CSV file saved to: {output_path}")


if __name__ == "__main__":
    input_file_path = ""
    output_file_path = ""
    api_key = ""

    process_csv(input_file_path, output_file_path, api_key)
