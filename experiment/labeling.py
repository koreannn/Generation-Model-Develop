from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


# Pandas 출력 설정
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)


def create_initial_labels(input_file, output_file, num_clusters=2):
    """TF-IDF와 K-means를 사용하여 초기 라벨을 생성합니다."""
    df = pd.read_csv(input_file)
    df.dropna(subset=["paragraph", "problems"], inplace=True)
    df["combined_text"] = df["paragraph"] + " " + df["problems"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["combined_text"])

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    df["target"] = kmeans.labels_
    final_columns = ["id", "paragraph", "problems", "question_plus", "target"]
    df[final_columns].to_csv(output_file, index=False)
    logger.info(f"초기 라벨링이 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")


def load_and_preprocess_data(file_path):
    """데이터를 로드하고 전처리합니다."""
    df = pd.read_csv(file_path)
    if "target" not in df.columns:
        raise ValueError("데이터셋에 'target' 열이 없습니다.")

    X = df[["paragraph", "problems"]].astype(str).agg(" ".join, axis=1)
    y = df["target"].astype(int)
    return df, X, y


def vectorize_text(X, max_features=5000):
    """텍스트 데이터를 벡터화합니다."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    return vectorizer.fit_transform(X)


def train_and_predict(X_vectorized, y, n_splits=5):
    """모델을 훈련하고 예측 확률을 반환합니다."""
    base_model = XGBClassifier(eval_metric="mlogloss", n_estimators=100)
    model = CleanLearning(base_model)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred_probs = np.zeros((len(y), len(np.unique(y))))

    for fold, (train_index, val_index) in enumerate(skf.split(X_vectorized, y), 1):
        logger.info(f"Fold {fold}/{n_splits}")
        X_train, X_val = X_vectorized[train_index], X_vectorized[val_index]
        y_train, _ = y[train_index], y[val_index]
        model.fit(X_train, y_train)
        pred_probs[val_index] = model.predict_proba(X_val)

    return pred_probs


def find_and_update_label_issues(df, y, pred_probs):
    """레이블 이슈를 찾고 데이터프레임을 업데이트합니다."""
    label_issues = find_label_issues(labels=y, pred_probs=pred_probs, return_indices_ranked_by="self_confidence")
    df["is_label_issue"] = False
    df.loc[label_issues, "is_label_issue"] = True
    df["suggested_label"] = np.argmax(pred_probs, axis=1)
    return df


def save_and_print_results(df, output_file):
    """결과를 저장하고 출력합니다."""
    final_columns = ["id", "paragraph", "problems", "question_plus", "target", "suggested_label", "is_label_issue"]
    df[final_columns].to_csv(output_file, index=False)

    logger.info("\nID와 제안된 레이블:")
    logger.info(df[["id", "suggested_label"]].to_string(index=False))

    logger.info("\n레이블 이슈 통계:")
    logger.info(df["is_label_issue"].value_counts(normalize=True))

    logger.info("\n원래 레이블과 제안된 레이블 비교:")
    logger.info(pd.crosstab(df["target"], df["suggested_label"]))


def main():
    initial_input_file = "../data/train.csv"
    initial_output_file = "../data/output_with_labels.csv"
    final_output_file = "../data/cleaned_output_with_labels_CL.csv"

    # 초기 라벨링 수행
    create_initial_labels(initial_input_file, initial_output_file)

    # 데이터 로드 및 전처리
    df, X, y = load_and_preprocess_data(initial_output_file)

    # 텍스트 벡터화
    X_vectorized = vectorize_text(X)

    # 모델 훈련 및 예측
    pred_probs = train_and_predict(X_vectorized, y)

    # 레이블 이슈 찾기 및 업데이트
    df = find_and_update_label_issues(df, y, pred_probs)

    # 결과 저장 및 출력
    save_and_print_results(df, final_output_file)


if __name__ == "__main__":
    main()
