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
    lr=2e-4,
    batch_size=8,
    checkpoint_steps=100,
    max_steps=1000,
)

ds = load_dataset("44David/SCoTD-deepseek-math-7B")

wandb.init(
    project="qwen-reasoning-0.5B", 
    tags=[config.train_type],
    job_type="train",
    config=config
)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


train_conf = SFTConfig(
    output_dir="./qwen_reasoning_500M",
    max_steps=config.max_steps,
    per_device_train_batch_size=config.batch_size,
    learning_rate=config.lr,
    logging_steps=10,
    save_steps=config.checkpoint_steps,
    eval_strategy="steps",
    eval_steps=50,
    report_to="wandb"
)

train_ds, eval_ds = ds.train_test_split(test_size=0.1).values()

trainer = SFTTrainer(
    model=model,
    args=train_conf,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

trainer.train()

