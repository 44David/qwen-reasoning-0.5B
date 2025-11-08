import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule_with_warmup
import random
import torch 
import torch.nn.functional as F 


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
    
    epochs = 3
    train_step = 0
    total_train_steps = epochs * len(dataloader) // gradient_accumulation_steps
    gradient_accumulation = 32
    gradient_accumulation_steps = 32 / batch_size 
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_constant_schedule_with_warmup(
        optim, 
        num_warmup_steps=total_train_steps // 10,
        num_training_steps=total_train_steps
    )
    
    wandb.init(
        project="qwen-reasoning-0.5B"
    )
    
    
    model.train()
    
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = batch.to(torch.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = F.cross_entropy(out.logits, batch["target"]) / gradient_accumulation
                loss.backward()
                
            if step%gradient_accumulation_steps == 0:
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/step": train_step,
                    "train/epoch": epoch,
                })
                
                optim.step()
                scheduler.step()
                optim.zero_grad()
                train_step += 1
                
            
    
main()