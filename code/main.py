import os

from data_loaders import DataLoader
from inference import InferenceModel
from loguru import logger
from model import ModelHandler
from trainer import CustomTrainer
from utils import (
    create_experiment_filename,
    load_config,
    load_env_file,
    log_config,
    set_logger,
    set_seed,
)
import wandb


def main():
    # env, config, log, seed 설정
    load_env_file()
    config = load_config()
    set_logger(log_file=config["log"]["file"], log_level=config["log"]["level"])
    set_seed()

    # wandb 설정
    exp_name = create_experiment_filename(config)
    wandb.init(
        config=config,
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=exp_name,
    )

    # wandb 실험명으로 config 갱신
    config["training"]["run_name"] = exp_name
    config["inference"]["output_path"] = os.path.join(config["inference"]["output_path"], exp_name + "_output.csv")
    log_config(config)

    try:
        # 모델 및 토크나이저 설정
        model_handler = ModelHandler(config["model"])
        model, tokenizer = model_handler.setup()

        # 학습용 데이터 처리
        data_processor = DataLoader(tokenizer, config["data"])
        train_dataset, eval_dataset = data_processor.prepare_datasets(is_train=True)
        test_dataset = data_processor.prepare_datasets(is_train=False)

        # 학습
        trainer = CustomTrainer(
            training_config=config["training"],
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trained_model = trainer.train()

        # 추론
        inferencer = InferenceModel(
            inference_config=config["inference"],
            model=trained_model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
        )
        inferencer.run_inference()

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        wandb.finish(exit_code=1)

    wandb.finish()


if __name__ == "__main__":
    main()
