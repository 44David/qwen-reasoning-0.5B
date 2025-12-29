import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from types import SimpleNamespace

device = "cuda"

# create config for wandb and sft
config = SimpleNamespace(
    train_type=f"qwen-reason-0.5B/sft",
    base_model="Qwen/Qwen2.5-0.5B",
    lr=1e-6,
    batch_size=16,
    gradient_accumulation_steps=4,
    checkpoint_steps=500,
    max_steps=5000,
    max_seq_length=2048
)

ds = load_dataset("qwedsacf/competition_math")

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


def format_prompt(example):
    return {
        "text": f"Problem: {example['problem']}\n\nSolution: {example['solution']}"
    }
    
ds = ds.map(format_prompt, remove_columns=ds['train'].column_names)

train_conf = SFTConfig(
    output_dir="./qwen_reasoning_500M",
    max_steps=config.max_steps,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.lr,
    lr_scheduler_type='cosine',
    warmup_steps=100,
    logging_steps=10,
    save_steps=config.checkpoint_steps,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=500,
    max_length=config.max_seq_length,
    bf16=True,
    dataset_text_field="text",
    max_grad_norm=1.0,
    gradient_checkpointing=True,
    report_to="wandb"
)

train_ds, eval_ds = ds['train'].train_test_split(test_size=0.1).values()

trainer = SFTTrainer(
    model=model,
    args=train_conf,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

trainer.train()

# save_model saves weights, config 
trainer.save_model("./qwen_reasoning_500M/model")

# saves tokenizer config
trainer.save_pretrained("./qwen_reasoning_500M/model")

wandb.finish()