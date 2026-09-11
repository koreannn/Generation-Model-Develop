import numpy as np
import torch
from tqdm import tqdm

from dpr_data import KorQuadDataset
from encoder import KobertBiEncoder
from indexers import DenseFlatIndexer


def build_passage_index(model, dataset, device, batch_size = 64):
    """
    - 검증셋의 고유 passage들을 임베딩해서 FAISS 인덱스로 변환
    - pid: 토큰화된 passage (중복 제거(같은 passage를 여러 질문이 공유하므로 한 번만 임베딩))
    """
    unique = {}
    for _, pid, p in dataset.tokenized_tuples:
        unique.setdefault(pid, p)
    
    pids = list(unique.keys())
    indexer = DenseFlatIndexer()
    indexer.init_index(768) # KoBERT hidden dim
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(pids), batch_size), desc = "passage 임베딩"):
            batch_pids = pids[i:i + batch_size]
            batch_tensors = [torch.tensor(unique[pid]) for pid in batch_pids]
            padded = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True, padding_value=dataset.pad_token_id)
            mask = (padded != dataset.pad_token_id).long().to(device)
            padded = padded.to(device)

            emb = model(padded, mask, "passage").cpu().numpy()
            indexer.index_data(list(zip(batch_pids, emb)))

    return indexer
            
            
            
def evaluate_topk_accuracy(model_ckpt_path, split = "validation", use_hard_negative = False, k_list = (1, 5, 20), device = "cpu"):
    model = KobertBiEncoder()
    model.load(model_ckpt_path)
    model.to(device)

    dataset = KorQuadDataset(split = split, use_hard_negative = use_hard_negative)
    indexer = build_passage_index(model, dataset, device)

    correct = {k: 0 for k in k_list}
    total = 0
    max_k = max(k_list)

    model.eval()
    with torch.no_grad():
        for q_tok, gold_pid, _ in tqdm(dataset.tokenized_tuples, desc="질문 평가"):
            q_tensor = torch.tensor(q_tok).unsqueeze(0).to(device)
            q_mask = torch.ones_like(q_tensor).to(device)
            q_emb = model(q_tensor, q_mask, "query").cpu().numpy()

            results = indexer.search_knn(q_emb, max_k)
            retrieved_pids = results[0][0]  # top-k passage id 리스트

            for k in k_list:
                if gold_pid in retrieved_pids[:k]:
                    correct[k] += 1
            total += 1

    for k in k_list:
        print(f"Top-{k} Accuracy: {correct[k] / total:.4f}")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    evaluate_topk_accuracy("./output/my_model.pt", split="validation", use_hard_negative = False, device = device)