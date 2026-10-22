"""
코퍼스 1단계: 한국어 위키백과 덤프를 문서 단위 jsonl로 정리하고 잡음 문서를 제거

- 입력: WikiExtractor --json 출력 (data/wiki/extracted/**/wiki_*)
    uvx --python 3.10 --from wikiextractor==3.0.6 wikiextractor \
        data/wiki/kowiki-20261001-pages-articles.xml.bz2 --json --processes 8 -b 100M -o data/wiki/extracted
    (wikiextractor 3.0.6은 Python 3.11 이상에서 정규식 오류로 실행되지 않아 3.10으로 따로 실행)
- 출력 (한 줄 = 문서 하나: doc_id, title, text)
    - data/wiki/kowiki_docs.jsonl: 본문이 있는 모든 문서
    - data/wiki/kowiki_corpus.jsonl: 검색 코퍼스. 아래 문서를 제외
        - 본문 100자 미만: 한 줄짜리 토막글, 동음이의어 문서 (WikiExtractor가 목록을 지워 안내 문장만 남음)
        - 제목에 "(동음이의)"가 있는 문서: 100자를 넘어도 다른 문서로 안내하는 내용뿐
- 주제 필터는 적용하지 않음: 문제 주제(한국사, 법, 사회, 대중문화, 세계사, 미국 법 등)가 넓고
  위키 분류 체계가 정돈되어 있지 않아, 걸러내면 필요한 인물·지명 문서가 빠질 위험이 큼
- 알려진 한계: WikiExtractor가 목록과 표를 지우므로 목록 문서의 표 내용은 코퍼스에 없음
"""

from collections import Counter
import html
import json
from pathlib import Path

from loguru import logger


WIKI_DIR = Path(__file__).resolve().parent.parent / "data" / "wiki"
EXTRACTED_DIR = WIKI_DIR / "extracted"
DOCS_PATH = WIKI_DIR / "kowiki_docs.jsonl"
CORPUS_PATH = WIKI_DIR / "kowiki_corpus.jsonl"

MIN_CHARS = 100
LENGTH_BINS = (100, 300, 1000, 3000, 10000)


def iter_extracted_docs():
    for path in sorted(EXTRACTED_DIR.glob("*/wiki_*")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def length_bin(n):
    for upper in LENGTH_BINS:
        if n < upper:
            return f"<{upper}"
    return f">={LENGTH_BINS[-1]}"


def get_exclude_reason(doc):
    if len(doc["text"]) < MIN_CHARS:
        return "short"
    if "(동음이의)" in doc["title"]:
        return "disambiguation"
    return None


def main():
    n_empty = 0
    n_docs = n_corpus = 0
    corpus_chars = 0
    bins = Counter()
    removed = Counter()

    with open(DOCS_PATH, "w", encoding="utf-8") as docs_out, open(CORPUS_PATH, "w", encoding="utf-8") as corpus_out:
        for raw in iter_extracted_docs():
            text = html.unescape(raw["text"]).strip()  # WikiExtractor가 남긴 &lt; 같은 HTML 엔티티 복원
            if not text:
                n_empty += 1
                continue

            doc = {"doc_id": raw["id"], "title": raw["title"], "text": text}
            line = json.dumps(doc, ensure_ascii=False) + "\n"
            docs_out.write(line)
            n_docs += 1
            bins[length_bin(len(text))] += 1

            reason = get_exclude_reason(doc)
            if reason:
                removed[reason] += 1
                continue
            corpus_out.write(line)
            n_corpus += 1
            corpus_chars += len(text)

    ordered = [f"<{upper}" for upper in LENGTH_BINS] + [f">={LENGTH_BINS[-1]}"]
    logger.info(f"본문 있는 문서 {n_docs}개 (빈 문서 {n_empty}개 제외): {DOCS_PATH}")
    logger.info(f"본문 길이(글자) 분포: {{{', '.join(f'{b}: {bins[b]}' for b in ordered)}}}")
    logger.info(f"코퍼스 문서 {n_corpus}개, 총 {corpus_chars / 1e6:.0f}M자 (제외 {dict(removed)}): {CORPUS_PATH}")


if __name__ == "__main__":
    main()
