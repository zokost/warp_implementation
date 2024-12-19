import wandb
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer
)
from trl import (
    RewardTrainer,
    RewardConfig,
)
import torch
import yaml
from prepare_data import prepare_dataset_for_reward
import os

def load_config(config_path='../configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    wandb.init(project="warp-reward", config=load_config())
    
    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(device)
    reward_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    reward_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1).to('cuda')
    
    reward_dataset = prepare_dataset_for_reward(reward_tokenizer, config)
    
    reward_config = RewardConfig(
        output_dir=config['reward']['output_dir'],
        max_length=config['reward']['max_length'],
        num_train_epochs=config['reward']['num_train_epochs'],
        per_device_train_batch_size=config['reward']['per_device_train_batch_size']
    )

    trainer = RewardTrainer(
        model=reward_model,
        train_dataset=reward_dataset,
        tokenizer=reward_tokenizer,
        args=reward_config
    )

    trainer.train()
    save_checkpoint(reward_model, step=1000)
    wandb.finish()