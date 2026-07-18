from ast import literal_eval
import re

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
            "documents": row.get("documents", None),
        }
        records.append(record)
    return data, records


def display_instance(left, right, record):
    with left:
        st.subheader("Paragraph")
        st.write(record["paragraph"])

        st.subheader("Question, Choices, Answer")
        st.markdown("#### Question:")
        st.write(record["question"])

        st.markdown("#### Choices:")
        for i, choice in enumerate(record["choices"], 1):
            if i == record["answer"]:
                st.markdown(f"<span style='color:red'>{i} : {choice}</span>", unsafe_allow_html=True)
            else:
                st.write(f"{i} : {choice}")

        st.markdown("#### Answer:")
        st.write(str(record["answer"]))

        st.subheader("Question Plus")
        st.write(str(record["question_plus"]))
    with right:
        st.subheader("Documents")
        result = re.split(r"(?=\[)", str(record["documents"]))
        for part in result:
            st.write(part)


def main(file_path="../data/train.csv"):
    st.set_page_config(layout="wide")
    st.title("CSV 데이터 인스턴스 뷰어")
    data, records = load_data(file_path)

    left, right = st.columns([0.5, 0.5])
    with left:
        instance_index = st.number_input("인스턴스 선택", min_value=0, max_value=len(data) - 1, value=0, step=1)

    display_instance(left, right, records[instance_index])

    with left:
        st.write(f"선택된 인스턴스 (인덱스 {instance_index}):")
        st.write(data.iloc[instance_index])


if __name__ == "__main__":
    main("../data/train_retrieve.csv")
