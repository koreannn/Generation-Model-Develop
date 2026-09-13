# from utils import get_passage_file
from glob import glob
import json
import logging
import math
import os
import pickle
import re
import typing
from typing import Iterator, List, Sized, Tuple

import torch
import numpy as np
from torch import tensor as T
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from retriever_elastic import ElasticsearchRetriever
from dotenv import load_dotenv

config_folder = os.path.join(os.path.dirname(__file__), "..", "..", "config")
load_dotenv(os.path.join(config_folder, ".env"))

def get_wiki_filepath(data_dir):
    return glob(f"{data_dir}/*/wiki_*")


def wiki_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # logger.debug(dataset)
    # dataset =
    overall_start = dataset.start
    overall_end = dataset.end
    split_size = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    # end_idx = min((worker_id+1) * split_size, len(dataset.data))
    dataset.start = overall_start + worker_id * split_size
    dataset.end = min(dataset.start + split_size, overall_end)  # index error 방지


def get_passage_file(p_id_list: typing.List[int]) -> str:
    """passage id를 받아서 해당되는 파일 이름을 반환합니다."""
    target_file = None
    p_id_max = max(p_id_list)
    p_id_min = min(p_id_list)
    for f in glob("processed_passages/*.p"):
        s, e = f.split("/")[1].split(".")[0].split("-")
        s, e = int(s), int(e)
        if p_id_min >= s and p_id_max <= e:
            target_file = f
    return target_file


# set logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


def korquad_collator(batch: List[Tuple], padding_value: int) -> Tuple[torch.Tensor]:
    """query, p_id, gold_passage를 batch로 반환합니다."""
    batch_q = pad_sequence([T(e[0]) for e in batch], batch_first=True, padding_value=padding_value)
    # logger.debug(batch_q.shape)
    batch_q_attn_mask = (batch_q != padding_value).long()
    # logger.debug(batch_q_attn_mask.shape)
    batch_p_id = T([e[1] for e in batch])[:, None]
    # logger.debug(batch_p_id.shape)
    batch_p = pad_sequence([T(e[2]) for e in batch], batch_first=True, padding_value=padding_value)
    # logger.debug(batch_p.shape)
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask)


def korquad_collator_hard_neg(batch, padding_value):
    batch_q = pad_sequence([T(e[0]) for e in batch], batch_first=True, padding_value=padding_value)
    batch_q_attn_mask = (batch_q != padding_value).long()
    batch_p_id = T([e[1] for e in batch])[:, None]
    batch_p = pad_sequence([T(e[2]) for e in batch], batch_first=True, padding_value=padding_value)
    batch_p_attn_mask = (batch_p != padding_value).long()
    batch_neg = pad_sequence([T(e[3]) for e in batch], batch_first=True, padding_value=padding_value)
    batch_neg_attn_mask = (batch_neg != padding_value).long()
    return (batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask, batch_neg, batch_neg_attn_mask)


class KorQuadSampler(torch.utils.data.BatchSampler):
    """in-batch negative학습을 위해 batch 내에 중복 answer를 갖지 않도록 batch를 구성합니다.
    sample 일부를 pass하기 때문에 전체 data 수보다 iteration을 통해 나오는 데이터 수가 몇십개 정도 적습니다."""

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        generator=None,
    ) -> None:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(data_source, replacement=False, generator=generator)
        else:
            sampler = torch.utils.data.SequentialSampler(data_source)
        super(KorQuadSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        sampled_p_id = []
        sampled_idx = []
        for idx in self.sampler:
            item = self.sampler.data_source[idx]
            if item[1] in sampled_p_id:
                continue  # 만일 같은 answer passage가 이미 뽑혔다면 pass
            sampled_idx.append(idx)
            sampled_p_id.append(item[1])
            if len(sampled_idx) >= self.batch_size:
                yield sampled_idx
                sampled_p_id = []
                sampled_idx = []
        if len(sampled_idx) > 0 and not self.drop_last:
            yield sampled_idx


class KorQuadDataset:
    def __init__(
        self, 
        split: str = "train",
        use_hard_negative: bool = False,
        index_name: str = "korquad-index",
        mining_batch_size: int = 500,
        top_k: int = 5,
        ):
        self.split = split
        self.use_hard_negative = use_hard_negative
        self.index_name = index_name
        self.mining_batch_size = mining_batch_size
        self.top_k = top_k
        self.data_tuples = []
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.pad_token_id = self.tokenizer.get_vocab()["[PAD]"]
        self.tokenized_tuples = None # self.load()를 통해 채워지는 값
        self.load()

    def __len__(self):
        return len(self.tokenized_tuples)
    
    @property
    def dataset(self) -> List[Tuple]:
        return self.tokenized_tuples
    
    def stat(self):
        """korquad 데이터셋의 스탯을 출력합니다."""
        raise NotImplementedError()
    
    def _mine_hard_negatives(self): # use_hard_negative = True일 경우에만 실행
        retriever = ElasticsearchRetriever(index_name = self.index_name)
        questions = [q for q, _, _ in self.data_tuples]
        gold_ids = [p_id for _, p_id, _ in self.data_tuples]
        
        hard_neg_texts = []
        for i in tqdm(range(0, len(questions), self.mining_batch_size), desc = "mining hard negatives (by ElasticSearch)"):
            batch_questions = questions[i: i + self.mining_batch_size]
            batch_gold_ids = gold_ids[i: i + self.mining_batch_size]
            batch_results = retriever.bulk_retrieve(batch_questions, top_k = self.top_k)
            
            for results, gold_id in zip(batch_results, batch_gold_ids):
                neg = next((r["text"] for r in results if r["id"] != str(gold_id)), results[0]["text"] if results else "")
                hard_neg_texts.append(neg)
        return hard_neg_texts
        

    def load(self):
        """데이터 전처리가 완료되었다면 load하고 그렇지 않으면 전처리를 수행합니다."""
        self.korquad_processed_path = f"korquad_{self.split}{'_hardneg' if self.use_hard_negative else ''}_processed.p"
        self._load_data()
        self._build_from_korquad_context()
        
        if os.path.exists(self.korquad_processed_path):
            logger.debug("preprocessed file already exists, loading...")
            with open(self.korquad_processed_path, "rb") as f:
                self.tokenized_tuples = pickle.load(f)
            logger.debug("successfully loaded tokenized_tuples into self.tokenized_tuples")

        else:
            
            if self.use_hard_negative:
                hard_neg_texts = self._mine_hard_negatives() 
                self.tokenized_tuples = [
                    (self.tokenizer.encode(q, max_length = 512, truncation = True), pid, self.tokenizer.encode(p, max_length = 512, truncation = True), self.tokenizer.encode(neg, max_length = 512, truncation = True))
                    for (q, pid, p), neg in tqdm(zip(self.data_tuples, hard_neg_texts), desc="tokenize")
                ]
            else:
                self.tokenized_tuples = [
                    (self.tokenizer.encode(q, max_length = 512, truncation = True), pid, self.tokenizer.encode(p, max_length = 512, truncation = True)) # 임시용 truncation
                    for q, pid, p in tqdm(self.data_tuples, desc="tokenize")
                ]
            self._save_processed_dataset()

    def _load_data(self):
        self.raw_dataset = load_dataset("KorQuAD/squad_kor_v1")[self.split]


    def _get_cand_ids(self, title):
        """미리 구축한 ko-wiki 데이터에서 해당 title에 맞는 id들을 가지고 옵니다."""
        refined_title = None
        ret = self.title_passage_map.get(title, None)
        if not ret:
            refined_title = re.sub(r"\(.*\)", "", title).strip()
            ret = self.title_passage_map.get(refined_title, None)
        return ret, refined_title

    def _build_from_korquad_context(self):
        self.context_to_id = {}
        self.context_to_title = {}
        for row in tqdm(self.raw_dataset, desc = "buliding passages from KorQuAD"):
            context = row["context"]
            if context not in self.context_to_id:
                self.context_to_id[context] = len(self.context_to_id)
                self.context_to_title[context] = row["title"]
            passage_ids = self.context_to_id[context]
            self.data_tuples.append((row["question"], passage_ids, context))
            
    def dump_passages_for_elasticsearch(self, output_path):
        passages = [None] * len(self.context_to_id)
        for context, passage_id in self.context_to_id.items():
            passages[passage_id] = {"title": self.context_to_title[context], "text": context}
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        with open(output_path, "w", encoding = "utf-8") as f:
            json.dump(passages, f, ensure_ascii = False)

    def _save_processed_dataset(self):
        """전처리한 데이터를 저장합니다."""
        with open(self.korquad_processed_path, "wb") as f:
            pickle.dump(self.tokenized_tuples, f)
        logger.debug(f"successfully saved self.tokenized_tuples into {self.korquad_processed_path}")


if __name__ == "__main__":
    ds = KorQuadDataset(split = "train", use_hard_negative = False)
    ds.dump_passages_for_elasticsearch("data/korquad_passages.json")