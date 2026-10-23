"""
위키 덤프에서 리다이렉트(넘겨주기) 목록을 뽑아 별칭 → 실제 문서 제목 사전을 만듦

- 입력: data/wiki/kowiki-*-pages-articles.xml.bz2 (원본 덤프)
- 출력: data/wiki/kowiki_redirects.json ({"국제 연합": "유엔", "해동통보": "고려의 화폐", ...})
- WikiExtractor는 리다이렉트 문서를 본문 없는 문서로 내보내고 대상 제목은 버려서 따로 뽑음
- 일반 문서(namespace 0)만 사용하고, "고려의 화폐#해동통보"처럼 섹션을 가리키면 문서 제목만 남김
- 정답 근거 후보 수집(build_gold_candidates.py)에서 LLM이 별칭으로 제안한 제목을 실제 문서로 연결할 때 사용
"""

import bz2
import html
import json
from pathlib import Path
import re

from loguru import logger


WIKI_DIR = Path(__file__).resolve().parent.parent / "data" / "wiki"
REDIRECTS_PATH = WIKI_DIR / "kowiki_redirects.json"

TITLE_RE = re.compile(r"<title>(.*?)</title>")
NS_RE = re.compile(r"<ns>(\d+)</ns>")
REDIRECT_RE = re.compile(r'<redirect title="(.*?)"')


def main():
    dump_path = sorted(WIKI_DIR.glob("kowiki-*-pages-articles.xml.bz2"))[-1]
    redirects = {}
    title = ns = None
    with bz2.open(dump_path, "rt", encoding="utf-8") as f:
        for line in f:
            if m := TITLE_RE.search(line):
                title = html.unescape(m.group(1))
            elif m := NS_RE.search(line):
                ns = m.group(1)
            elif (m := REDIRECT_RE.search(line)) and ns == "0":
                redirects[title] = html.unescape(m.group(1)).split("#")[0].strip()

    with open(REDIRECTS_PATH, "w", encoding="utf-8") as f:
        json.dump(redirects, f, ensure_ascii=False)
    logger.info(f"리다이렉트 {len(redirects)}개 저장: {REDIRECTS_PATH} (덤프: {dump_path.name})")


if __name__ == "__main__":
    main()
