import os
import sys

# 현재 코드가 있는 디렉토리 기준으로 상위 디렉토리를 `sys.path`에 추가
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from copy import deepcopy
import logging
from typing import Tuple

from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator, korquad_collator_hard_neg
from encoder import KobertBiEncoder
import numpy as np
import torch
from tqdm import tqdm
import transformers

from utils import load_config


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()  # get root logger


class Trainer:
    """basic trainer"""

    def __init__(
        self,
        model,
        device,
        train_dataset,
        valid_dataset,
        num_epoch: int,
        batch_size: int,
        lr: float,
        betas: Tuple[float],
        num_warmup_steps: int,
        num_training_steps: int,
        valid_every: int,
        best_val_ckpt_path: str,
        use_hard_negative: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )
        self.use_hard_negative = use_hard_negative
        collate_fn = korquad_collator_hard_neg if use_hard_negative else korquad_collator
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset.dataset,
            batch_sampler=KorQuadSampler(train_dataset.dataset, batch_size=batch_size, drop_last=False),
            collate_fn=lambda x: collate_fn(x, padding_value=train_dataset.pad_token_id),
            num_workers=4,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset.dataset,
            batch_sampler=KorQuadSampler(valid_dataset.dataset, batch_size=batch_size, drop_last=False),
            collate_fn=lambda x: collate_fn(x, padding_value=valid_dataset.pad_token_id),
            num_workers=4,
        )

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.valid_every = valid_every
        self.lr = lr
        self.betas = betas
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.best_val_ckpt_path = best_val_ckpt_path
        self.best_val_optim_path = best_val_ckpt_path.split(".pt")[0] + "_optim.pt"

        self.start_ep = 1
        self.start_step = 1

    def ibn_loss(self, pred: torch.FloatTensor):
        """in-batch negative를 활용한 batch의 loss를 계산합니다.
        pred : bsz x bsz 또는 bsz x bsz*2의 logit 값을 가짐. 후자는 hard negative를 포함하는 경우.
        """
        bsz = pred.size(0)
        target = torch.arange(bsz).to(self.device)  # 주대각선이 answer
        return torch.nn.functional.cross_entropy(pred, target)

    def batch_acc(self, pred: torch.FloatTensor):
        """batch 내의 accuracy를 계산합니다."""
        bsz = pred.size(0)
        target = torch.arange(bsz)  # 주대각선이 answer
        return (pred.detach().cpu().max(1).indices == target).sum().float() / bsz

    def fit(self):
        """모델을 학습합니다."""
        # wandb.init(
        #     project="personal",
        #     entity="gayean01",
        #     config={
        #         "batch_size": self.batch_size,
        #         "lr": self.lr,
        #         "betas": self.betas,
        #         "num_warmup_steps": self.num_warmup_steps,
        #         "num_training_steps": self.num_training_steps,
        #         "valid_every": self.valid_every,
        #     },
        # )
        logger.debug("start training")
        self.model.train()  # 학습모드
        global_step_cnt = 0
        prev_best = None
        for ep in range(self.start_ep, self.num_epoch + 1):
            for step, batch in enumerate(tqdm(self.train_loader, desc=f"epoch {ep} batch"), 1):
                if ep == self.start_ep and step < self.start_step:
                    continue  # 중간부터 학습시키는 경우 해당 지점까지 복원
                
                self.model.train()  # 학습 모드
                if self.use_hard_negative:
                    q, q_mask, _, p, p_mask, neg, neg_mask = batch
                    neg, neg_mask = neg.to(self.device), neg_mask.to(self.device)
                else:
                    q, q_mask, _, p, p_mask = batch
                q, q_mask, p, p_mask = q.to(self.device), q_mask.to(self.device), p.to(self.device), p_mask.to(self.device)
                q_emb = self.model(q, q_mask, "query")  # bsz x bert_dim
                p_emb = self.model(p, p_mask, "passage")  # bsz x bert_dim
                if self.use_hard_negative:
                    neg_emb = self.model(neg, neg_mask, "passage")
                    p_all_emb = torch.cat([p_emb, neg_emb], dim = 0)
                else:
                    p_all_emb = p_emb
                
                pred = torch.matmul(q_emb, p_all_emb.T) # bsz x bsz
                loss = self.ibn_loss(pred)
                acc = self.batch_acc(pred)
                global_step_cnt += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                log = {
                    "epoch": ep,
                    "step": step,
                    "global_step": global_step_cnt,
                    "train_step_loss": loss.cpu().item(),
                    "current_lr": float(self.scheduler.get_last_lr()[0]),  # parameter group 1개이므로
                    "step_acc": acc,
                }
                if global_step_cnt % self.valid_every == 0:
                    eval_dict = self.evaluate()
                    log.update(eval_dict)
                    if prev_best is None or eval_dict["valid_loss"] < prev_best:  # best val loss인 경우 저장
                        # self.model.checkpoint(self.best_val_ckpt_path)
                        self.save_training_state(log)
                # wandb.log(log)

    def evaluate(self):
        """모델을 평가합니다."""
        self.model.eval()  # 평가 모드
        loss_list = []
        sample_cnt = 0
        valid_acc = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                if self.use_hard_negative:
                    q, q_mask, _, p, p_mask, neg, neg_mask = batch
                    neg, neg_mask = neg.to(self.device), neg_mask.to(self.device)
                else:
                    q, q_mask, _, p, p_mask = batch
                q, q_mask, p, p_mask = q.to(self.device), q_mask.to(self.device), p.to(self.device), p_mask.to(self.device)
                q_emb = self.model(q, q_mask, "query")  # bsz x bert_dim
                p_emb = self.model(p, p_mask, "passage")  # bsz x bert_dim
                if self.use_hard_negative:
                    neg_emb = self.model(neg, neg_mask, "passage")
                    p_all_emb = torch.cat([p_emb, neg_emb], dim = 0)
                else:
                    p_all_emb = p_emb
                
                pred = torch.matmul(q_emb, p_all_emb.T) # bsz x bsz
                loss = self.ibn_loss(pred)
                step_acc = self.batch_acc(pred)

                bsz = q.size(0)
                sample_cnt += bsz
                valid_acc += step_acc * bsz
                loss_list.append(loss.cpu().item() * bsz)
        valid_loss = np.array(loss_list).sum() / float(sample_cnt)
        valid_acc = valid_acc / float(sample_cnt)

        # 콘솔에 출력
        logger.info(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")
        return {
            "valid_loss": np.array(loss_list).sum() / float(sample_cnt),
            "valid_acc": valid_acc / float(sample_cnt),
        }

    def save_training_state(self, log_dict: dict) -> None:
        """모델, optimizer와 기타 정보를 저장합니다"""
        self.model.checkpoint(self.best_val_ckpt_path)
        training_state = {
            "optimizer_state": deepcopy(self.optimizer.state_dict()),
            "scheduler_state": deepcopy(self.scheduler.state_dict()),
        }
        training_state.update(log_dict)
        torch.save(training_state, self.best_val_optim_path)
        logger.debug(f"saved optimizer/scheduler state into {self.best_val_optim_path}")

    def load_training_state(self) -> None:
        """모델, optimizer와 기타 정보를 로드합니다"""
        self.model.load(self.best_val_ckpt_path)
        training_state = torch.load(self.best_val_optim_path)
        logger.debug(f"loaded optimizer/scheduler state from {self.best_val_optim_path}")
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        self.scheduler.load_state_dict(training_state["scheduler_state"])
        self.start_ep = training_state["epoch"]
        self.start_step = training_state["step"]
        logger.debug(f"resume training from epoch {self.start_ep} / step {self.start_step}")


# 모델 존재 여부 확인 함수
def check_if_model_exists(model_path: str):
    """모델 체크포인트가 존재하는지 확인하는 함수"""
    return os.path.exists(model_path)


# 메인 실행
if __name__ == "__main__":
    # 모델 경로 설정
    config = load_config()["bi_encoder"]
    model_path = config["output_path"]["best_val_ckpt_path"]

    # 모델이 없으면 학습 시작
    if not check_if_model_exists(model_path):
        logger.info(f"모델 '{model_path}'이 존재하지 않습니다. 학습을 시작합니다.")

        # 학습을 위한 준비
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = KobertBiEncoder()
        train_dataset = KorQuadDataset(
            split = config["data"]["train_split"],
            use_hard_negative = config["data"]["use_hard_negative"],
            index_name = config["elasticsearch"]["index_name"],
            mining_batch_size = config["data"]["hard_negative"]["mining_batch_size"],
            top_k = config["data"]["hard_negative"]["top_k"]
        )
        valid_dataset = KorQuadDataset(
            split = config["data"]["valid_split"],
            use_hard_negative = config["data"]["use_hard_negative"],
            index_name = config["elasticsearch"]["index_name"],
            mining_batch_size = config["data"]["hard_negative"]["mining_batch_size"],
            top_k = config["data"]["hard_negative"]["top_k"]
        )
        
        # h_param
        batch_size = config["params"]["batch_size"]
        steps_per_epoch = len(train_dataset) // batch_size
        num_training_steps = config["params"]["num_epochs"] * steps_per_epoch
        num_warmup_steps = int(num_training_steps * config["params"]["warmup_ratio"])

        my_trainer = Trainer(
            model = model,
            device = device,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            num_epoch = config["params"]["num_epochs"],
            batch_size = batch_size,
            lr = config["params"]["lr"],
            betas = tuple(config["params"]["betas"]),
            num_warmup_steps = num_warmup_steps,
            num_training_steps = num_training_steps,
            valid_every = config["params"]["valid_every"],
            best_val_ckpt_path = model_path,
            use_hard_negative = config["data"]["use_hard_negative"],
        )

        # 학습 수행
        my_trainer.fit()
        eval_dict = my_trainer.evaluate()
        logger.info(eval_dict)

        # 모델 저장 디렉토리 생성 및 저장
        os.makedirs("output", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"학습 완료. 모델이 '{model_path}'에 저장되었습니다.")
    else:
        logger.info(f"모델 '{model_path}'이 이미 존재합니다. 학습을 건너뜁니다.")
