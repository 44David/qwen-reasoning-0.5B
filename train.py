import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule_with_warmup
import random
import torch 

class SCoTDDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data['train'][idx]
        thinking_trace = random.choice(sample["thinking_traces"])
        
        sample = f"{sample} {thinking_trace}"
        tokens = self.tokenizer(
            sample,
            max_length=self.max_length,
        ) 
        input_ids = tokens['input_ids'].squeeze(0)
        return {
            'input_ids': input_ids[:-1],
            'targets': input_ids[1:],
        }
    
        
def main():
    ds = load_dataset("44David/SCoTD-deepseek-math-7B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    batch_size = 8
    dataloader = DataLoader(
        dataset=SCoTDDataset(ds, tokenizer), 
        batch_size=8,
        shuffle=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        dtype=torch.bfloat16
    )
    
    layers_to_freeze = 18
    for param in model.parameters(): param.requires_grad = False
    for param in model.lm_head.parameters(): param.requires_grad = False
    for param in model.model.layers[layers_to_freeze:].parameters(): param.requires_grad = True
    
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {trainable_params_count:,}")
    
    
    
main()