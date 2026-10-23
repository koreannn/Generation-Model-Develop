"""
코퍼스 2단계: 위키 문서를 검색 단위 청크로 나눔

- 입력: data/wiki/kowiki_corpus.jsonl (build_wiki_corpus.py 출력)
- 출력: data/wiki/kowiki_chunks.jsonl (한 줄 = 청크 하나)
    chunk_id, doc_id, title, section, order, text
    - text는 "제목 > 섹션" 헤더 한 줄 + 본문 (섹션이 없는 도입부는 "제목"만)
    - doc_id, order로 같은 문서의 앞뒤 청크를 다시 찾을 수 있음
- 나누는 방식
    - 문단을 순서대로 묶어 본문 TARGET_CHARS(500자)를 채우고, MAX_CHARS(800자)는 넘기지 않음
    - 새 섹션이 시작될 때 현재 청크가 SECTION_MERGE_CHARS(250자) 이상이면 청크를 끊고,
      그보다 짧으면 이어 붙이되 본문에 "섹션명." 줄을 넣어 경계를 남김
      (한두 문장짜리 섹션이 많아 섹션마다 끊으면 헤더가 본문보다 긴 청크가 대량으로 생김)
    - 헤더와 section에는 청크가 시작된 섹션을 씀
    - MAX_CHARS를 넘는 긴 문단만 문장 단위로 나누고, 이어지는 청크에 앞 청크의 마지막 문장을 겹쳐 넣음
      (문단 경계는 내용 경계라 겹칠 필요가 없고, 문단 중간을 자를 때만 문맥이 끊기기 때문)
    - 버리는 내용은 없음 (짧은 섹션도 사실 한 줄을 담고 있는 경우가 많음)
- 크기 근거: 문단 길이 중앙값 141자, 90%가 353자 이하, 800자 초과는 0.4%
  500자는 KoBERT 토크나이저 기준 약 290토큰으로 인코더 최대 길이(512) 안에 헤더까지 들어감
- 알려진 한계: WikiExtractor가 섹션 단계(==, ===)를 지워 하위 섹션도 같은 단계로 보고 바로 위 제목만 헤더로 씀
"""

from collections import Counter
import json
from pathlib import Path
import re

from loguru import logger


WIKI_DIR = Path(__file__).resolve().parent.parent / "data" / "wiki"
CORPUS_PATH = WIKI_DIR / "kowiki_corpus.jsonl"
CHUNKS_PATH = WIKI_DIR / "kowiki_chunks.jsonl"

TARGET_CHARS = 500
MAX_CHARS = 800
SECTION_MERGE_CHARS = 250
HEADING_MAX_CHARS = 30
LENGTH_BINS = (100, 300, 500, 800)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def is_heading(line):
    """WikiExtractor는 섹션 제목을 "생애." 처럼 마침표를 붙인 한 줄로 남김"""
    return len(line) <= HEADING_MAX_CHARS and line.endswith(".") and not line.endswith("다.") and ". " not in line


def split_sections(text):
    """[(섹션 제목 또는 None, [문단, ...]), ...]"""
    sections = [(None, [])]
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if is_heading(line):
            sections.append((line[:-1], []))
        else:
            sections[-1][1].append(line)
    return [(heading, paragraphs) for heading, paragraphs in sections if paragraphs]


def split_long_paragraph(paragraph):
    """문장 단위로 TARGET_CHARS까지 묶고, 다음 조각은 앞 조각의 마지막 문장부터 시작"""
    sentences = []
    for sentence in SENTENCE_END.split(paragraph):
        # 한 문장이 MAX_CHARS를 넘으면 글자 수로 자름
        sentences += [sentence[i : i + MAX_CHARS] for i in range(0, len(sentence), MAX_CHARS)]

    pieces = []
    start = 0
    while start < len(sentences):
        end = start + 1
        length = len(sentences[start])
        while end < len(sentences) and length < TARGET_CHARS and length + 1 + len(sentences[end]) <= MAX_CHARS:
            length += 1 + len(sentences[end])
            end += 1
        pieces.append(" ".join(sentences[start:end]))
        if end == len(sentences):
            break
        start = end - 1 if end - 1 > start else end  # 마지막 문장을 겹치되, 한 문장짜리 조각이면 그대로 진행
    return pieces


def pack_document(text):
    """(시작 섹션, 본문, 긴 문단을 나눈 조각인지) 리스트"""
    bodies = []
    section = None  # 현재 청크가 시작된 섹션
    current = []

    def flush(next_section):
        nonlocal section, current
        if current:
            bodies.append((section, "\n".join(current), False))
        section, current = next_section, []

    for heading, paragraphs in split_sections(text):
        if not current or len("\n".join(current)) >= SECTION_MERGE_CHARS:
            flush(heading)
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) > MAX_CHARS:
                flush(heading)
                bodies += [(heading, piece, True) for piece in split_long_paragraph(paragraph)]
                continue
            # 짧은 앞 청크에 이어 붙이는 섹션의 첫 문단 앞에는 섹션명을 넣어 경계를 남김
            line = f"{heading}.\n{paragraph}" if i == 0 and current else paragraph
            length = len("\n".join(current))
            if current and (length >= TARGET_CHARS or length + 1 + len(line) > MAX_CHARS):
                flush(heading)
                line = paragraph
            current.append(line)
    flush(None)
    return bodies


def chunk_document(doc, stats):
    chunks = []
    for order, (section, body, from_split) in enumerate(pack_document(doc["text"])):
        header = f"{doc['title']} > {section}" if section else doc["title"]
        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}-{order}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "section": section,
                "order": order,
                "text": f"{header}\n{body}",
            }
        )
        stats["from_split"] += from_split
        stats["body_chars"] += len(body)
        stats["bins"][length_bin(len(body))] += 1
    return chunks


def length_bin(n):
    for upper in LENGTH_BINS:
        if n <= upper:
            return f"<={upper}"
    return f">{LENGTH_BINS[-1]}"


def main():
    stats = {"from_split": 0, "body_chars": 0, "bins": Counter()}
    n_docs = n_chunks = 0

    with open(CORPUS_PATH, encoding="utf-8") as f, open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for line in f:
            doc = json.loads(line)
            chunks = chunk_document(doc, stats)
            n_docs += 1
            n_chunks += len(chunks)
            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            if n_docs % 100000 == 0:
                logger.info(f"진행 {n_docs}문서, {n_chunks}청크")

    ordered = [f"<={upper}" for upper in LENGTH_BINS] + [f">{LENGTH_BINS[-1]}"]
    distribution = ", ".join(f"{b}: {stats['bins'][b]}" for b in ordered)
    logger.info(f"문서 {n_docs}개 → 청크 {n_chunks}개 (문서당 {n_chunks / n_docs:.1f}개)")
    logger.info(f"본문 평균 {stats['body_chars'] / n_chunks:.0f}자, 분포: {{{distribution}}}")
    logger.info(f"긴 문단을 문장 단위로 나눈 청크 {stats['from_split']}개")
    logger.info(f"저장 완료: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
