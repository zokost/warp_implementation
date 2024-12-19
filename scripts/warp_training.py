import wandb
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader
import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from prepare_data import create_prompts
import yaml
import os
from datasets import load_dataset, Dataset, DatasetDict


def load_config(config_path='../configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

import torch
import torch.nn.functional as F

def slerp(initial_state_dict, model_0, model_1, interpolation_factor, eps=1e-6):

    # Flatten the differences between model_0, model_1, and the initial model
    delta_0_flatten = torch.cat([param_0.flatten() - param_init.flatten() for param_0, param_init in zip(model_0.values(), initial_state_dict.values())], 0)
    delta_1_flatten = torch.cat([param_1.flatten() - param_init.flatten() for param_1, param_init in zip(model_1.values(), initial_state_dict.values())], 0)

    # Compute the angle between the two flattened task vectors using cosine similarity
    angle = torch.acos(F.cosine_similarity(delta_0_flatten, delta_1_flatten, dim=0).clamp(-1 + eps, 1 - eps)) + eps
    
    # Compute the interpolation coefficients based on the angle
    coef_0 = (torch.sin((1 - interpolation_factor) * angle) / torch.sin(angle)).item()
    coef_1 = (torch.sin(interpolation_factor * angle) / torch.sin(angle)).item()


    for name, param_init in initial_state_dict.items():
        initial_state_dict[name] = (param_init.data.detach().clone().mul_(1 - coef_0 - coef_1)
                                    .add_(model_0[name].data, alpha=coef_0)
                                    .add_(model_1[name].data, alpha=coef_1))

    
    return initial_state_dict


def liti(theta_init, theta_slerp, nu):
    averaged_state_dict = {}
    for param_name in theta_init.keys():
        averaged_state_dict[param_name] = (1 - nu) * theta_init[param_name] + nu * theta_slerp[param_name]
    
    return averaged_state_dict

def train_warp(config):
    I = config['training_params']['iterations']
    T = config['training_params']['training_steps']
    M = config['training_params']['rl_runs']
    mu = float(config['training_params']['ema_update_rate'])
    nu = float(config['training_params']['interpolation_factor'])
    beta = float(config['training_params']['kl_coefficient'])
    batch_size = int(config['training_params']['batch_size'])
    learning_rate = float(config['training_params']['learning_rate'])
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    reward_model = DistilBertForSequenceClassification.from_pretrained(config['model_paths']['reward_model']).to(device)
    reward_tokenizer = DistilBertTokenizer.from_pretrained(config['model_paths']['reward_tokenizer'])

    sft_model = GPT2LMHeadModel.from_pretrained(config['model_paths']['sft_model']).to(device)
    sft_model_tokenizer = GPT2Tokenizer.from_pretrained(config['model_paths']['sft_model'])
    sft_model_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    sft_model_tokenizer.pad_token = sft_model_tokenizer.eos_token
    sft_model.generation_config.pad_token_id = sft_model_tokenizer.pad_token_id
    
    prev_model = GPT2LMHeadModel.from_pretrained(config['model_paths']['sft_model']).to(device)
    
    theta_init = sft_model.state_dict()
    optimizer = torch.optim.AdamW(sft_model.parameters(), lr=1e-5)
    
    train_prompts = create_prompts()
    
    device = 'cuda'
    for i in range(I):
        theta_list = []
        
        theta_m_list = []
        theta_m_ema_list = []
        for m in range(M):
            theta_m = {k: v.clone() for k, v in theta_init.items()}
            theta_m_ema = {k: v.clone() for k, v in theta_init.items()}
            
            losses = []
            for prompt in tqdm(train_prompts[:T]):
                optimizer.zero_grad()
                
                inputs = sft_model_tokenizer.encode(prompt, return_tensors='pt').to(device)
                outputs = sft_model.generate(inputs, max_length=65, num_return_sequences=1)
                generated_prompt = sft_model_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                inputs = sft_model_tokenizer.encode(generated_prompt, return_tensors="pt").to(device)
                outputs = sft_model(inputs, labels=inputs)
                loss = outputs.loss
                
                reward_inputs = reward_tokenizer.encode(generated_prompt, return_tensors="pt").to(device)
                rewards = reward_model(reward_inputs).logits
                
                logits = sft_model(inputs).logits
                logits = F.log_softmax(logits, dim=-1)            
            
                logits_ema = prev_model(inputs).logits
                logits_ema = F.softmax(logits_ema, dim=-1)
                
                kl_reward = F.kl_div(logits, logits_ema)
                
                final_loss = loss * rewards[0] - beta * kl_reward * loss
                
                final_loss_abs = torch.abs(final_loss)
                wandb.log({
                    "loss": final_loss_abs.item(),
                    # "reward": rewards[0].item(),
                    # "kl_reward": kl_reward.item()
                })
                
                final_loss.backward()
                optimizer.step()
                
                for param_name, param_value in theta_m.items():
                    theta_m_ema[param_name] = (1 - mu) * theta_m_ema[param_name] + mu * param_value
                prev_model.load_state_dict(theta_m_ema)
            
            theta_m_list.append(theta_m)
            theta_m_ema_list.append(theta_m_ema)
            
        theta_slerp = slerp(theta_init, theta_m_list[0], theta_m_list[1], 0.5)    
        for k in theta_init.keys():
            theta_init[k] = (1 - nu) * theta_init[k] + nu * theta_slerp[k]

    weights = liti(sft_model.state_dict(), theta_slerp, 1 / 2)

    sft_model.load_state_dict(weights)
    
    
    sft_model.save_pretrained(config['model_paths']['output_model'])
    
    artifact = wandb.Artifact("final-model", type="model")
    artifact.add_file(os.path.join(config['model_paths']['output_model'], "pytorch_model.bin"))
    wandb.log_artifact(artifact)
    
    return sft_model


if __name__ == "__main__":
    config = load_config()
    
    wandb.init(project="warp-training", config=config)
    
    train_warp(config)
    
    wandb.finish()