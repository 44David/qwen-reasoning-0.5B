# qwen-reasoning-0.5B

By using processes like Symbollic Chain-of-Thought Distilliation (SCoTD): [arXiv 2306.14050](https://arxiv.org/abs/2306.14050), supervised fine tuning and RL algorithms like GRPO, 
we can add chain-of-thought thinking traces to models smaller in size than their teacher models, from which the thinking traces data is collected. 

Allowing chain-of-thought reasoning in models that otherwise do not have this capability. 

You can find the dataset I created for SCoTD here: https://huggingface.co/datasets/44David/gsm8k-reasoning-traces
