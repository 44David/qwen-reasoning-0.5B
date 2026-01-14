import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import gc
import re
import glob

# clear runpod cache
torch.cuda.empty_cache()
gc.collect()

import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, interleave_datasets
from trl import SFTConfig, SFTTrainer
from types import SimpleNamespace

device = "cuda"

# create config for wandb and sft
config = SimpleNamespace(
    train_type=f"qwen-reason-0.5B/sft-mixed",
    base_model="Qwen/Qwen2.5-0.5B",
    lr=5e-5,
    batch_size=6,
    gradient_accumulation_steps=2,
    checkpoint_steps=500,
    max_steps=5000,
    max_seq_length=3072
)

gsm8k_traces = load_dataset("44David/gsm8k-reasoning-traces")
alpaca_chat = load_dataset("tatsu-lab/alpaca")

wandb.init(
    project="qwen-reasoning-0.5B", 
    tags=[config.train_type],
    job_type="train",
    config=config
)


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


def format_math(example):
    # Remove <<>> annotations and #### marker
    answer_clean = re.sub(r'<<[^>]+>>', '', example['answer'])
    answer_clean = answer_clean.split('####')[0].strip()
    
    messages = [
        {"role": "user", "content": example['problem']},
        {"role": "assistant", "content": f"<think>{example['reasoning']}</think>\n\n{answer_clean}"}
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
    }
    
def format_alpaca(example):
    instruction = example['instruction']
    
    # some examples have an extra line of context
    if example.get('input') and example['input'].strip():
        instruction = f"{instruction}\n{example['input']}"
    
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": example['output']} 
    ]
    
    return {
        "text": tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
    }
    

gsm8k_formatted = gsm8k_traces.map(format_math, remove_columns=gsm8k_traces['train'].column_names)
alpaca_formatted = alpaca_chat.map(format_alpaca, remove_columns=alpaca_chat['train'].column_names)


train_conf = SFTConfig(
    output_dir="./qwen_reasoning_500M_v2",
    max_steps=config.max_steps,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.lr,
    lr_scheduler_type='cosine',
    warmup_steps=100,
    logging_steps=10,
    save_steps=config.checkpoint_steps,
    save_total_limit=None, # keep all 10 checkpoints, (55GB total)
    eval_strategy="steps",
    eval_steps=500,
    max_length=config.max_seq_length,
    bf16=True,
    dataset_text_field="text",
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    report_to="wandb"
)

mixed_train = interleave_datasets(
    [gsm8k_formatted['train'], alpaca_formatted['train']],
    probabilities = [0.7, 0.3],
    seed = 42
)

mixed_eval = interleave_datasets(
    [gsm8k_formatted.get('test', gsm8k_formatted['train'].train_test_split(test_size=0.1)['test']),
     alpaca_formatted['train'].train_test_split(test_size=0.1)['test']],
    probabilities=[0.7, 0.3],
    seed=42
)

trainer = SFTTrainer(
    model=model,
    args=train_conf,
    train_dataset=mixed_train,
    eval_dataset=mixed_eval,
    processing_class=tokenizer,
)


checkpoints = glob.glob("./qwen_reasoning_500M_v2/checkpoint-*")

if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    resume_path = latest_checkpoint
    
else:
    resume_path = None

trainer.train(resume_from_checkpoint=resume_path)

wandb.finish()