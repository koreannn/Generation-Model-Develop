"""
Oracle 실험 추론

- config.yaml의 oracle 섹션만 바꿔서 실행 (model_name, inference_mode, split)
    - no_rag: 문제만
    - oracle_chunk: 정답 근거가 들어 있는 청크를 그대로 넣음
    - oracle_sentence: 정답 근거 문장만 넣음
    - rag: 검색기가 찾은 상위 k개 청크를 넣음 (oracle 섹션의 retrieval: query_type, top_k)
- 답은 생성하지 않고, 첫 토큰을 선택지 번호 토큰으로만 제한해 고름
- 출력: {output_dir}/{split}/{모델 이름}/{inference_mode}.jsonl (한 줄 = 문제 하나)
    - rag는 rag_{query_type}_k{top_k}.jsonl
"""

import json
from pathlib import Path
import random

from bm25_search import BM25Retriever, retrieve
from loguru import logger
from vllm import LLM, SamplingParams
import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"


def load_eval_set(path, split, limit=None): # gold인 문서만 반환하되 split 모드에 따라 dev/test중 반환
    """oracle_eval_set.jsonl에서 split이 같고 gold_status가 gold인 문제만 리스트로 반환"""
    records = []
    with open(path, 'r', encoding = 'utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['split'] == split and record['gold_status'] == "gold":
                records.append(record)
    return records[:limit] if limit else records


def load_chunks(path, chunk_ids): # 전체 위키 청킹된 코퍼스 중 Oracle 청크에 해당하는 코퍼스만 가져오기
    chunks = {}
    with open(path, 'r', encoding = 'utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            if chunk['chunk_id'] in chunk_ids:
                chunks[chunk['chunk_id']] = chunk['text']
    return chunks



# ---------- 모드별 참고 문서 만들기 ----------
# 반환값: 프롬프트에 넣을 참고 문서 문자열, 없으면 None
# rng: 문제마다 따로 만든 random.Random (순서 섞기용)



"""Oracle 평가 모드별 참조 방식"""
def context_no_rag(record, chunks, rng):
    return None


# 2. 청크를 넣을 경우
def context_oracle_chunk(record, chunks, rng):
    chunk_ids = list(dict.fromkeys(e["chunk_id"] for e in record["evidence"]))
    rng.shuffle(chunk_ids)
    return "\n\n".join(f"[문서 {i}]\n{chunks[cid]}" for i, cid in enumerate(chunk_ids, 1))
    

# 3. 문장만 넣을 경우
def context_oracle_sentence(record, chunks, rng):
    sentences = []
    for evidence in record["evidence"]:
        sentences.append(evidence["sentence"])
    rng.shuffle(sentences)
    sentence = '\n'.join(f"- {line}" for line in sentences)
    return sentence


def context_rag(record, chunks, rng):
    # [기능] 검색기가 찾은 상위 k개 청크를 프롬프트에 넣을 참고 문서 문자열로 만듦
    #        검색은 main()에서 미리 해서 record["retrieved"]에 chunk_id 리스트(검색 순위 순)로 넣어 둠
    #        chunks: chunk_id → 청크 텍스트 (main()에서 필요한 청크만 불러옴)
    # TODO
    # - record["retrieved"] 순서 그대로 사용 (섞지 않음: 실제 서비스에서도 검색 순위대로 넣으므로)
    #   rng는 다른 모드와 함수 모양을 맞추려고 받기만 하고 쓰지 않음
    # - 형식은 context_oracle_chunk와 같게: "[문서 1]\n{텍스트}\n\n[문서 2]\n..."
    chunks_list = [f"[문서 {i}]\n{chunks[chunk_id]}" for i, chunk_id in enumerate(record["retrieved"], 1)]
    context = '\n\n'.join(chunks_list)
    return context


def needed_chunk_ids(records, mode):
    # [기능] 프롬프트에 들어갈 청크 id를 모두 모음
    #        청크 파일(1.5GB) 전체를 메모리에 올리지 않고 이 id들만 load_chunks로 불러오기 위함
    # TODO
    # - oracle_chunk: 모든 record의 evidence에 있는 chunk_id
    # - rag: 모든 record의 record["retrieved"]에 있는 chunk_id
    # - 집합(set)으로 반환
    chunk_ids = set()
    for record in records:
        if mode == "oracle_chunk":
            chunk_ids.update(e["chunk_id"] for e in record["evidence"]) # 라벨링된 근거 청크
        elif mode == "rag":
            chunk_ids.update(record["retrieved"]) # 검색기가 찾은 상위 k개 청크
    return chunk_ids

CONTEXT_BUILDERS = {
    "no_rag": context_no_rag,
    "oracle_chunk": context_oracle_chunk,
    "oracle_sentence": context_oracle_sentence,
    "rag": context_rag,
}


def build_prompt(record, context, tokenizer):
    parts = []
    if record["paragraph"]:
        parts.append(f"지문:\n{record['paragraph']}")
    parts.append(f"질문:\n{record['question']}")
    parts.append("선택지:\n" + "\n".join(f"{i}. {choice}" for i, choice in enumerate(record["choices"], 1)))
    
    if context is not None:  # no_rag일 경우 참고 문서 칸을 통째로 뺌
        parts.append(f"참고 문서:\n{context}")
    parts.append(f"1~{len(record['choices'])} 중 정답 번호 하나만 답하세요.")
    content = "\n\n".join(parts)
        
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize = False,
        add_generation_prompt = True,
        enable_thinking = False,
    )
    return prompt + "정답: "


def choice_token_ids(tokenizer, n_choices):
    """"1"~"n" 각각의 토큰 id 리스트. 숫자 하나가 토큰 하나인지 확인"""
    ids = []
    for i in range(1, n_choices + 1):
        token_ids = tokenizer.encode(str(i), add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"'{i}'가 토큰 하나가 아님: {token_ids}")
        ids.append(token_ids[0])
    return ids


def predict(llm, prompts, records, tokenizer):
    allowed = [choice_token_ids(tokenizer, len(record["choices"])) for record in records]
    params = [
        SamplingParams(max_tokens=1, temperature=0.0, allowed_token_ids=ids)  # 번호 토큰 하나만, 가장 확률 높은 것
        for ids in allowed
    ]
    outputs = llm.generate(prompts, params)  # prompts[i]에 params[i]가 적용되고 결과도 같은 순서로 나옴

    preds = []
    for output, ids in zip(outputs, allowed):
        token_id = output.outputs[0].token_ids[0]  # 생성된 첫(이자 유일한) 토큰
        preds.append(ids.index(token_id) + 1)  # "1"의 토큰이면 0번째 → 예측 번호 1
    return preds


def output_path(config):
    model_dir = config["model_name"].replace("/", "__")
    name = config["inference_mode"]
    
    if name == "rag":  # 검색 설정을 바꿔 가며 비교하므로 설정을 파일 이름에 넣어 덮어쓰기 방지
        name = f"rag_{config['retrieval']['query_type']}_k{config['retrieval']['top_k']}"
    return Path(config["data"]["output_dir"]) / config["split"] / model_dir / f"{name}.jsonl"


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)["oracle"]
    mode = config["inference_mode"]
    if mode not in CONTEXT_BUILDERS:
        raise ValueError(f"알 수 없는 inference_mode: {mode} (가능: {sorted(CONTEXT_BUILDERS)})")

    records = load_eval_set(config["data"]["eval_set_path"], config["split"], config.get("limit"))
    if mode == "rag":
        retriever = BM25Retriever()
        retrieval = config["retrieval"]
        for record in records:
            record["retrieved"] = retrieve(retriever, record, retrieval["query_type"], retrieval["top_k"])
        logger.info(f"검색 완료: {retrieval['query_type']}, top {retrieval['top_k']}")

    chunks = {}
    if mode in ("oracle_chunk", "rag"):
        chunks = load_chunks(config["data"]["chunks_path"], needed_chunk_ids(records, mode))
    logger.info(f"{config['split']} {len(records)}문제, 모델 {config['model_name']}, 모드 {mode}")

    llm = LLM(model=config["model_name"], seed=config["seed"], **config["vllm"])
    tokenizer = llm.get_tokenizer()

    build_context = CONTEXT_BUILDERS[mode]
    prompts = []
    for record in records:
        rng = random.Random(f"{config['seed']}-{record['qid']}")  # 문제별 고정 시드: 처리 순서와 무관하게 같은 순서
        prompts.append(build_prompt(record, build_context(record, chunks, rng), tokenizer))
    logger.info(f"프롬프트 예시:\n{prompts[0]}")  # 끝부분이 "정답:" 바로 뒤에서 끝나는지 확인

    preds = predict(llm, prompts, records, tokenizer)

    path = output_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record, pred in zip(records, preds):
            row = {"qid": record["qid"], "pred": pred, "answer": record["answer"], "correct": pred == record["answer"]}
            if mode == "rag":  # 나중에 "근거를 찾았을 때와 못 찾았을 때 정답률" 분석용
                row["retrieved"] = record["retrieved"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    accuracy = sum(p == r["answer"] for p, r in zip(preds, records)) / len(records)
    logger.info(f"정답률 {accuracy:.4f} ({len(records)}문제), 저장: {path}")


if __name__ == "__main__":
    main()