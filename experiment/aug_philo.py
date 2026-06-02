import os
import warnings

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd


def parse_output(output):
    lines = output.split("\n")
    data_list = []
    data = {"지문": "", "문제": "", "보기": "", "정답": "", "해설": ""}
    current_key = None

    for line in lines:
        line = line.strip()
        if line.startswith("지문:"):
            if data["지문"]:
                data_list.append(data.copy())
                data = {"지문": "", "문제": "", "보기": "", "정답": "", "해설": ""}
            current_key = "지문"
            data["지문"] = line.replace("지문:", "").strip()
        elif line.startswith("문제:"):
            current_key = "문제"
            data["문제"] = line.replace("문제:", "").strip()
        elif line.startswith("보기:"):
            current_key = "보기"
            data["보기"] = line.replace("보기:", "").strip()
        elif line.startswith("정답:"):
            current_key = "정답"
            data["정답"] = line.replace("정답:", "").strip()
        elif line.startswith("해설:"):
            current_key = "해설"
            data["해설"] = line.replace("해설:", "").strip()
        elif current_key:
            data[current_key] += " " + line

    if data["지문"]:
        data_list.append(data)

    return data_list


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["OPENAI_API_KEY"] = ""

    # 사용할 LLM 모델 설정
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.9)

    # 프롬프트 템플릿 설정
    prompt_template = """"""

    prompt = PromptTemplate(input_variables=[], template=prompt_template)

    response = llm.invoke(prompt.format())
    output = response.content

    parsed_data_list = parse_output(output)

    df = pd.DataFrame(parsed_data_list)

    csv_filename = "philosophy_questions.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
