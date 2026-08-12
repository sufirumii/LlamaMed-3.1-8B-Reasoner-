# LlamaMed-3.1-8B-Reasoner

A Llama-3.1-8B fine-tune trained to reason step by step through medical multiple-choice questions before answering, using chain-of-thought supervision from the ReasonMed dataset.

Not a medical device. This is a research checkpoint for exploring medical-reasoning fine-tunes. Nothing it produces should inform a real clinical decision without a licensed clinician in the loop.

## Model details

| | |
|---|---|
| Base model | [unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit) |
| Dataset | [lingshu-medical-mllm/ReasonMed](https://huggingface.co/datasets/lingshu-medical-mllm/ReasonMed) — first 10,000 samples |
| Method | QLoRA (4-bit), rank 16, via [Unsloth](https://github.com/unslothai/unsloth) |
| Fine-tuned weights | [Rumiii/LlamaMed-3.1-8B-Reasoner](https://huggingface.co/Rumiii/LlamaMed-3.1-8B-Reasoner) |
| License | Apache 2.0 |

## Dataset

Training data comes from **ReasonMed**, a chain-of-thought medical reasoning dataset hosted on the Hugging Face Hub:

- Dataset page: https://huggingface.co/datasets/lingshu-medical-mllm/ReasonMed
- File used: `ReasonMed.json`
- Only the first 10,000 samples of the dataset are used, streamed directly from the Hub (no full local download required)
- Each sample follows an `instruction` / `output` schema, where the chain-of-thought reasoning is embedded directly in `output`

You can preview or download the dataset yourself via the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset(
    "lingshu-medical-mllm/ReasonMed",
    data_files="ReasonMed.json",
    split="train",
)
```

## Fine-tuning

`scripts/finetune.py` is the exact script used to produce this model. It runs on a single GPU (developed and tested on a Kaggle T4) using Unsloth for memory-efficient QLoRA fine-tuning.

### Requirements

```
unsloth
torch
transformers
datasets
trl
```

Install order matters — Unsloth's install has to happen after its dependencies are already present. See `requirements-finetune.txt` for exact pinned versions.

### Configuration summary

| Setting | Value |
|---|---|
| Max sequence length | 1024 |
| Quantization | 4-bit |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Gradient checkpointing | Unsloth (memory-efficient) |
| Per-device batch size | 2 |
| Gradient accumulation steps | 8 (effective batch size 16) |
| Max steps | 600 (~1 epoch over 10,000 samples at effective batch 16) |
| Learning rate | 2e-4 |
| LR scheduler | linear |
| Optimizer | adamw_8bit |
| Weight decay | 0.01 |
| Seed | 3407 |

### Script

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"                      # forces single-GPU, avoids the device-map error
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduces fragmentation-related OOMs

import gc
import torch

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

free, total = torch.cuda.mem_get_info()
print(f"Free before load: {free / 1e9:.2f} GB / {total / 1e9:.2f} GB")

# ---------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------
from unsloth import FastLanguageModel

max_seq_length = 1024   # kept at 1024 rather than 2048 -- this was the key fix for OOM at this scale
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = {"": 0},
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",   # Unsloth's memory-efficient checkpointing -- confirm the
                                                # "Will smartly offload gradients" line prints after this
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ---------------------------------------------------------------------
# 2. Load ReasonMed, streamed, first 10K
# ---------------------------------------------------------------------
from datasets import load_dataset, Dataset

streamed = load_dataset(
    "lingshu-medical-mllm/ReasonMed",
    data_files="ReasonMed.json",
    split="train",
    streaming=True,
)

rows = []
for i, row in enumerate(streamed):
    rows.append(row)
    if i + 1 >= 10000:
        break

raw_dataset = Dataset.from_list(rows)
print(f"Loaded {len(raw_dataset)} samples")

# ---------------------------------------------------------------------
# 3. Format using the model's own chat template
#    (confirmed real schema: instruction / output -- CoT is embedded in output)
# ---------------------------------------------------------------------
def formatting_prompts_func(examples):
    texts = []
    n = len(examples["instruction"])
    for i in range(n):
        messages = [
            {"role": "user", "content": examples["instruction"][i]},
            {"role": "assistant", "content": examples["output"][i]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

dataset = raw_dataset.map(formatting_prompts_func, batched=True)
print(f"Formatted {len(dataset)} training examples")

# ---------------------------------------------------------------------
# 4. Lightweight adapter-only checkpoint callback
#    (avoids the SFTConfig pickling crash from the built-in checkpointing)
# ---------------------------------------------------------------------
from transformers import TrainerCallback

class SaveAdapterCallback(TrainerCallback):
    def __init__(self, save_every=25, output_dir="/kaggle/working/checkpoints"):
        self.save_every = save_every
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_every == 0:
            path = f"{self.output_dir}/adapter-step-{state.global_step}"
            kwargs["model"].save_pretrained(path)
            tok = kwargs.get("processing_class") or kwargs.get("tokenizer")
            if tok:
                tok.save_pretrained(path)
            print(f"Saved adapter checkpoint at step {state.global_step} -> {path}")
        return control

# ---------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    callbacks = [SaveAdapterCallback(save_every=25, output_dir="/kaggle/working/checkpoints")],
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,        # effective batch size 16
        warmup_steps = 10,
        max_steps = 600,                         # one epoch over 10,000 at effective batch 16
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        save_strategy = "no",                    # avoids the SFTConfig pickling bug entirely
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "/kaggle/working/checkpoints",
        report_to = "none",
        average_tokens_across_devices = False,   # fixes the 'int' object has no attribute 'mean' crash
    ),
)

print("Starting training...")
trainer.train()

# ---------------------------------------------------------------------
# 6. Save final adapter
# ---------------------------------------------------------------------
model.save_pretrained("/kaggle/working/final_model")
tokenizer.save_pretrained("/kaggle/working/final_model")
print("Saved to /kaggle/working/final_model")
```

### Running it

```bash
git clone https://github.com/sufirumii/LlamaMed-3.1-8B-Reasoner
cd LlamaMed-3.1-8B-Reasoner
pip install -r requirements-finetune.txt
python scripts/finetune.py
```

The script streams the dataset directly from the Hub, so no manual dataset download step is required. Adjust `max_steps`, `per_device_train_batch_size`, and `gradient_accumulation_steps` to fit your own GPU's memory budget.

## Inference

The fine-tuned weights are published on the Hugging Face Hub and can be loaded directly:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Rumiii/LlamaMed-3.1-8B-Reasoner",
    max_seq_length = 1024,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
```

## Repository layout

```
scripts/
  finetune.py               training script used to produce this model
requirements-finetune.txt   pinned dependencies for fine-tuning
LICENSE
README.md
```

## Disclaimer

This is a research checkpoint, not a validated clinical tool. It has not been reviewed by a regulatory body and must not inform real medical decisions. Outputs should always be checked by a licensed clinician before being acted on.

## License

Apache 2.0
