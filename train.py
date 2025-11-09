import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, get_constant_schedule_with_warmup
import random
import torch 
import torch.nn.functional as F 
from types import SimpleNamespace
from torchmetrics import Accuracy
import tqdm 

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
    

def compute_loss(x, y):
    return F.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))

def validate(model, validate_dataloader):
    model.eval()
    acc = Accuracy()
    
    for step, batch in enumerate(tqdm(validate_dataloader)):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
            loss = compute_loss(out.logits, batch["target"])
            
        acc.update(out.logits, batch["target"])
            
    wandb.log({
        "validate/loss": loss.item(),
        "validate/accuracy": acc.compute()
    })
    
    model.train()


def main():
    ds = load_dataset("44David/SCoTD-deepseek-math-7B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    batch_size = 8
    train_ds_size = int(0.8 * len(ds))
    validate_ds_size = len(ds) - train_ds_size 
    
    train_ds, validate_ds = random_split(ds, [train_ds_size, validate_ds_size])
    
    train_dataloader = DataLoader(
        dataset=SCoTDDataset(train_ds, tokenizer), 
        batch_size=batch_size,
        shuffle=False
    )
    
    validate_dataloader = DataLoader(
        dataset=SCoTDDataset(validate_ds, tokenizer), 
        batch_size=batch_size,
        shuffle=False
    )

    
    # create config 
    config = SimpleNamespace(
        train_type=f"qwen-reason-0.5B/scotd",
        base_model="Qwen/Qwen2.5-0.5B",
        precision="bf16",
        lr=2e-4,
        epochs=3,
        batch_size=8,
        gradient_accumulation=32,
        gradient_accumulation_steps=32//config.batch_size,   
        total_train_steps=config.epochs * len(train_dataloader) // config.gradient_accumulation_steps,
        layer_freeze=18,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch.bfloat16
    )
    
    for param in model.parameters(): param.requires_grad = False
    for param in model.lm_head.parameters(): param.requires_grad = False
    for param in model.model.layers[config.layer_freeze:].parameters(): param.requires_grad = True
    
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {trainable_params_count:,}")
    
    train_step = 0
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_constant_schedule_with_warmup(
        optim, 
        num_warmup_steps=config.total_train_steps // 10,
        num_training_steps=config.total_train_steps
    )
    
    wandb.init(
        project="qwen-reasoning-0.5B", 
        tags=[config.train_type],
        job_type="train",
        config=config
    )
    
    acc = Accuracy()
    model.train()
    progress_bar = tqdm(total=config.total_train_steps)
    for epoch in range(config.epochs):
        for step, batch in enumerate(train_dataloader):
            batch = batch.to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = compute_loss(out.logits, batch["target"]) / config.gradient_accumulation
                loss.backward()
                
            if step%config.gradient_accumulation_steps == 0:
                wandb.log({
                    "train/loss": loss.item() * config.gradient_accumulation_steps,
                    "train/accuracy": acc.update(out.logits, batch["targets"]),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/step": train_step,
                    "train/epoch": epoch,
                })
                
                optim.step()
                scheduler.step()
                optim.zero_grad()
                train_step += 1
                progress_bar.update(1)
                
        # validate model after each epoch
        validate(model, validate_dataloader)
    
    progress_bar.close()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    })
    
    wandb.finish()
    
main()