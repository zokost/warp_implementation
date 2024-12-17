from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import numpy as np
import yaml
from tqdm import tqdm

def create_prompts(test=False):
    try:
        dataset = load_dataset("stanfordnlp/imdb")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def generate_prompts(dataset, num_samples=None):
        if num_samples:
            sampled_data = np.random.choice(dataset, num_samples, replace=False)
        else:
            sampled_data = dataset

        prompts = [comment["text"][:25] for comment in tqdm(sampled_data)]
        return prompts

    if test:
        prompts = generate_prompts(test_dataset, num_samples=200)
    else:
        prompts = generate_prompts(train_dataset)

    return prompts

def generate_test(tokenizer, config, test=False):
    dataset = load_dataset('imdb')
    prompts = []
    
    if not test:
        data = dataset['train']
        max_prompts = config['dataset']['max_prompts']
    else:
        data = dataset['test']
        max_prompts = config['dataset']['test_max_prompts']  
    
    for i, item in enumerate(data):
        if i >= max_prompts:
            break
        text = item['text']
        tokens = tokenizer.tokenize(text)
        truncated_tokens = tokens[:np.random.randint(config['dataset']['min_tokens'], config['dataset']['max_tokens'] + 1)]
        prompt = tokenizer.convert_tokens_to_string(truncated_tokens)
        prompts.append(prompt)
    
    return prompts