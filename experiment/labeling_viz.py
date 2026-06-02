from ast import literal_eval

import pandas as pd
import streamlit as st


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
            "question_plus": problems.get("question_plus", None),
            "target": problems.get("target", None),
            "suggested_label": problems.get("suggested_label", None),
            "is_label_issue": problems.get("is_label_issue", None),
        }
        records.append(record)
    return data, records


def display_instance(record):
    st.subheader("Paragraph")
    st.write(record["paragraph"])

    st.subheader("Question, Choices, Answer")
    st.markdown("#### Question:")
    st.write(record["question"])

    st.markdown("#### Choices:")
    for i, choice in enumerate(record["choices"], 1):
        st.write(f"{i} : {choice}")

    st.markdown("#### Answer:")
    st.write(str(record["answer"]))

    st.subheader("Question Plus")
    st.write(str(record["question_plus"]))


def main():
    st.title("CSV 데이터 인스턴스 뷰어")

    data, records = load_data("../data/cleaned_output_with_labels_CL.csv")

    # 1055번째 인덱스를 기준으로 데이터 분할
    split_index = 792
    before_split = data.iloc[:split_index]
    after_split = data.iloc[split_index:]

    # 1055 이전과 이후의 suggested_label 개수 계산
    label_counts = {
        "Before 1380": {
            "Label 0": (before_split["suggested_label"] == 0).sum(),
            "Not Label 0": (before_split["suggested_label"] != 0).sum(),
        },
        "After 1380": {
            "Label 1": (after_split["suggested_label"] == 1).sum(),
            "Not Label 1": (after_split["suggested_label"] != 1).sum(),
        },
    }

    # 결과를 데이터프레임으로 변환
    label_counts_df = pd.DataFrame(label_counts)

    # 라벨 개수 출력
    st.subheader("Suggested Label 개수")
    st.write(label_counts_df)

    # Before Split에서 suggested_label이 1인 행 필터링
    before_split_label_1 = before_split[before_split["suggested_label"] == 1]

    # After Split에서 suggested_label이 1인 행 필터링
    after_split_label_1 = after_split[after_split["suggested_label"] == 0]

    # 결과 출력
    st.subheader("Before 1380에서 suggested_label이 1인 인스턴스")
    st.write(before_split_label_1)

    st.subheader("After 1380에서 suggested_label이 0인 인스턴스")
    st.write(after_split_label_1)

    # 인스턴스 선택 기능 추가
    instance_index = st.number_input("인스턴스 선택", min_value=0, max_value=len(data) - 1, value=0, step=1)

    st.write(f"선택된 인스턴스 (인덱스 {instance_index}):")
    st.write(data.iloc[instance_index])

    display_instance(records[instance_index])


if __name__ == "__main__":
    main()
