import json
import os
import re
from typing import Dict, List, Optional
import warnings

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, ElasticsearchWarning
from loguru import logger


# ElasticsearchWarning 무시
warnings.filterwarnings("ignore", category=ElasticsearchWarning)


class ElasticsearchRetriever:
    def __init__(
        self,
        index_name: str = "wiki-index",
        data_path: Optional[str] = None,
        setting_path: Optional[str] = None,
        doc_filename: Optional[str] = None,
    ) -> None:
        self.index_name = index_name
        self.client = self._connect_elasticsearch(
            os.getenv("ELASTICSEARCH_URL"), os.getenv("ELASTICSEARCH_ID"), os.getenv("ELASTICSEARCH_PW")
        )

        # 데이터셋 로드 및 인덱스 초기화
        if not self.client.indices.exists(index=self.index_name):
            if data_path and setting_path and doc_filename:
                docs = self._load_dataset(os.path.join(data_path, doc_filename))
                self._initialize_index(setting_path)
                self._insert_documents(docs)
            else:
                raise ValueError(f"존재하지 않는 인덱스: {index_name}")

    def _connect_elasticsearch(self, url: str, id: str, pw: str) -> Elasticsearch:
        """ElasticSearch 클라이언트 연결"""
        es = Elasticsearch(
            url,
            basic_auth=(id, pw),
            request_timeout=30,
            max_retries=10,
            retry_on_timeout=True,
            verify_certs=False,
        )
        logger.info(f"Elasticsearch 연결 상태: {es.ping()}")
        return es

    def _load_dataset(self, doc_filename) -> Dict:
        """문서 데이터셋 로드"""
        with open(doc_filename, "r", encoding="utf-8") as f:
            return json.load(f)

    def _initialize_index(self, setting_path) -> None:
        """인덱스 생성 및 설정"""
        with open(setting_path, "r") as f:
            setting = json.load(f)
        self.client.indices.create(index=self.index_name, body=setting)
        logger.info("인덱스 생성 완료")

    def _delete_index(self):
        if not self.client.indices.exists(index=self.index_name):
            logger.info("Index doesn't exist.")
            return

        self.client.indices.delete(index=self.index_name)
        logger.info("Index deletion has been completed")

    def _insert_documents(self, docs) -> None:
        """문서 데이터 bulk 삽입"""

        def _preprocess(text):
            text = re.sub(r"\n", " ", text)
            text = re.sub(r"\\n", " ", text)
            text = re.sub(r"#", " ", text)
            text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
            return text

        bulk_data = []
        for i, doc in enumerate(docs):
            # bulk 작업을 위한 메타데이터
            bulk_data.append({"index": {"_index": self.index_name, "_id": i}})
            # 실제 문서 데이터
            bulk_data.append({"title": doc["title"], "text": _preprocess(doc["text"])})

            # 1000개 단위로 벌크 삽입 수행
            if (i + 1) % 1000 == 0:
                try:
                    response = self.client.bulk(body=bulk_data)
                    if response["errors"]:
                        logger.warning(f"{i+1}번째 벌크 삽입 중 일부 오류 발생")
                    bulk_data = []  # 벌크 데이터 초기화
                    logger.info(f"{i+1}개 문서 벌크 삽입 완료")
                except Exception as e:
                    logger.error(f"벌크 삽입 실패 (인덱스: {i}): {e}")
                    bulk_data = []  # 오류 발생 시에도 데이터 초기화

        # 남은 데이터 처리
        if bulk_data:
            try:
                response = self.client.bulk(body=bulk_data)
                if response["errors"]:
                    logger.warning("마지막 벌크 삽입 중 일부 오류 발생")
            except Exception as e:
                logger.error(f"마지막 벌크 삽입 실패: {e}")

        # 최종 문서 수 확인
        n_records = self.client.count(index=self.index_name)["count"]
        logger.info(f"총 {n_records}개 문서 삽입 완료")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """단일 쿼리에 대한 검색 수행"""
        query_body = {"query": {"bool": {"must": [{"match": {"text": query}}]}}}

        response = self.client.search(index=self.index_name, body=query_body, size=top_k)

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "text": hit["_source"]["text"],
                "score": hit["_score"],
            })
        return results

    def bulk_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Dict]]:
        """여러 쿼리에 대한 일괄 검색 수행 (msearch API 사용)"""
        logger.info(f"{len(queries)}개 쿼리 일괄 검색")

        # msearch API를 위한 bulk 쿼리 준비
        bulk_query = []
        for query in queries:
            # 메타데이터 라인
            bulk_query.append({"index": self.index_name})
            # 쿼리 라인
            bulk_query.append({"query": {"bool": {"must": [{"match": {"text": query}}]}}, "size": top_k})

        try:
            # msearch API 호출
            response = self.client.msearch(body=bulk_query)

            # 결과 처리
            results = []
            for response_item in response["responses"]:
                query_results = []
                if not response_item.get("error"):
                    for hit in response_item["hits"]["hits"]:
                        query_results.append({
                                        "id": hit["_id"],
                                        "text": hit["_source"]["text"],
                                        "score": hit["_score"],
                                    })
                results.append(query_results)

            logger.info(f"{len(queries)}개 쿼리 일괄 검색 완료")
            return results

        except Exception as e:
            logger.error(f"Bulk search 실패: {e}")
            return [[] for _ in queries]  # 에러 발생 시 빈 결과 반환


if __name__ == "__main__":
    config_folder = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    load_dotenv(os.path.join(config_folder, ".env"))

    retriever = ElasticsearchRetriever(
        index_name = "korquad-index",
        data_path = "data/",
        setting_path="config/elastic_setting.json",
        doc_filename="korquad_passages.json",
    )

    # 새로운 문서 추가 삽입시에만 사용
    if False:
        current_count = retriever.client.count(index=retriever.index_name)["count"]

        logger.info("새로운 문서 추가 시작")
        with open("new_wiki.json", "r", encoding="utf-8") as f:
            new_docs = json.load(f)
        retriever._insert_documents(new_docs)

        # 문서 추가 확인
        new_count = retriever.client.count(index=retriever.index_name)["count"]
        logger.info(f"문서 추가 완료: {current_count} -> {new_count} ({new_count-current_count}개 추가)")

    # 문서 검색 테스트
    print(retriever.client.count(index="korquad-index"))  # count가 9606인지 확인
    results = retriever.retrieve("세종대왕이 한글을 창제한 이유는?", top_k=3)

    for i, result in enumerate(results, 1):
        logger.debug(f"\n검색 결과 {i}")
        logger.debug(f"점수: {result['score']:.4f}")
        logger.debug(f"내용: {result['text'][:200]}...")
