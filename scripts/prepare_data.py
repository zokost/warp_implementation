from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import numpy as np
import yaml
from tqdm import tqdm



from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import numpy as np
import yaml

class PairsDataset(Dataset):
    def __init__(self, positive_comments, negative_comments, tokenizer, max_length=512):
        self.positive_comments = positive_comments
        self.negative_comments = negative_comments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.positive_comments) * len(self.negative_comments)

    def __getitem__(self, idx):
        pos_idx = idx // len(self.negative_comments)
        neg_idx = idx % len(self.negative_comments)
        pos_comment = self.positive_comments[pos_idx]
        neg_comment = self.negative_comments[neg_idx]
        
        pos_inputs = self.tokenizer(pos_comment, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        neg_inputs = self.tokenizer(neg_comment, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        return {
            'input_ids_chosen': pos_inputs['input_ids'].squeeze(),
            'attention_mask_chosen': pos_inputs['attention_mask'].squeeze(),
            'input_ids_rejected': neg_inputs['input_ids'].squeeze(),
            'attention_mask_rejected': neg_inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(1, dtype=torch.float)  # Награда для каждой пары
        }

def prepare_dataset_for_reward(tokenizer, config):
    dataset = load_dataset('imdb')
    positive_comments = [item['text'] for item in dataset['train'] if item['label'] == 1][:config['dataset']['num_positive_samples']]
    negative_comments = [item['text'] for item in dataset['train'] if item['label'] == 0][:config['dataset']['num_negative_samples']]
    pairs_dataset = PairsDataset(positive_comments, negative_comments, tokenizer, max_length=config['dataset']['max_length'])
    return pairs_dataset

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