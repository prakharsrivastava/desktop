# LoRA & QLoRA — Complete Mastery Guide

> Sources: Krish Naik (voice transcript + GitHub: krishnaik06/Finetuning-LLM) · Patrik Szepesi (GitHub: patrikszepesi/qlora-course) · Research Papers (Hu et al. 2021, Dettmers et al. 2023)

---

## Table of Contents

1. [Why Fine-Tuning Exists — The Problem](#1-why-fine-tuning-exists)
2. [Full Parameter Fine-Tuning & Its Challenges](#2-full-parameter-fine-tuning)
3. [What Is Quantization](#3-what-is-quantization)
4. [LoRA — Low-Rank Adaptation](#4-lora)
5. [QLoRA — Quantized LoRA](#5-qlora)
6. [Key Hyperparameters Explained](#6-key-hyperparameters)
7. [The Full Pipeline: Step-by-Step Code (Patrik + Krish)](#7-full-pipeline-code)
8. [Training Arguments Deep Dive](#8-training-arguments)
9. [MLflow Integration for Tracking](#9-mlflow-tracking)
10. [Pros & Cons](#10-pros-and-cons)
11. [Alternatives to LoRA / QLoRA](#11-alternatives)
12. [Interview Cheat Sheet](#12-interview-cheat-sheet)

---

## 1. Why Fine-Tuning Exists

A **pre-trained LLM** (like GPT-4, LLaMA 2, Mistral 7B) is trained on massive internet-scale data and learns general language understanding. When you give it a domain-specific question, it gives a generic answer.

```
Input: Domain document → LLM → Generic answer
Goal:  Input: Domain document → Fine-tuned LLM → Specific domain answer
```

### Three types of fine-tuning (Krish Naik)

| Type | What happens | Example |
|---|---|---|
| Full parameter fine-tuning | All weights updated | ChatGPT, Claude |
| Domain-specific fine-tuning | Fine-tune for one domain | Finance GPT, Medical GPT |
| Task-specific fine-tuning | Fine-tune for one task | Text-to-SQL, Document Q&A |

---

## 2. Full Parameter Fine-Tuning

When you fine-tune a model completely, **every single weight gets updated**.

- LLaMA 2 7B → 7 billion weights to update
- GPT-3 → 175 billion weights to update

### The problem

- Requires enormous GPU VRAM (hundreds of GBs)
- Extremely expensive on cloud (AWS, GCP)
- Slow — billions of gradient calculations
- Impractical on a single GPU or consumer hardware

This is why **LoRA and QLoRA** were invented.

---

## 3. What Is Quantization

### Definition (Krish Naik)

> "Quantization = conversion from a higher memory format to a lower memory format."

### Memory formats you must know

| Format | Bits | Name | Range |
|---|---|---|---|
| FP32 | 32 | Full Precision / Single Precision | High |
| FP16 | 16 | Half Precision | Medium |
| BF16 | 16 | Brain Float (better range than FP16) | Medium |
| INT8 | 8 | 8-bit integer | Low |
| INT4 / NF4 | 4 | 4-bit / NormalFloat4 | Very Low |

Comparison Table
Precision	Bits	Bytes	70B Model Memory
FP32	32	4	280 GB
FP16	16	2	140 GB
INT8	8	1	70 GB
4-bit	4	0.5	35 GB
### How FP32 is stored in memory (Krish breakdown)

Easy Comparison Table
Mode	Restaurant Story	AI Meaning
PTQ	Compress recipes after training chefs	Quantize after training
QAT	Train chefs using compressed recipes	Simulate quantization during training
PTQ	Faster deployment	Slight accuracy drop
QAT	Better adaptation	Better accuracy


A 32-bit float like `7.32` uses memory as:
- 1 bit → sign (positive/negative)
- 8 bits → exponent (the `7` part)
- 23 bits → mantissa/fraction (the `.32` part)

FP16 uses: 1 bit sign + 5 bits exponent + 10 bits mantissa.

### Why quantize?

- 70B parameter model at FP32 needs ~280 GB RAM — impossible on normal GPU
- Converting to INT8: 70 GB — more manageable
- Converting to INT4 (QLoRA): ~35 GB — fits on 2-3 consumer GPUs
- Faster inference — smaller numbers = less computation

### The trade-off

Quantization introduces a small **loss of accuracy** (information loss when compressing 32 bits to 8 or 4). This is managed by calibration and quantization-aware techniques.

### Two quantization modes

| Mode | When used | Description |
|---|---|---|
| Post-Training Quantization (PTQ) | Inference only | Quantize after training is done. Fast but slight accuracy drop. |
| Quantization-Aware Training (QAT) | Fine-tuning | Simulate quantization during training so the model adapts. Preserves accuracy. |

**Important (Krish Naik):** All fine-tuning techniques (LoRA, QLoRA) use **Quantization-Aware Training**, not PTQ — so accuracy loss is controlled.

### Calibration

The mathematical process of converting FP32 → INT8. Uses a **scale factor**:

```
scale = (x_max - x_min) / (Q_max - Q_min)
quantized_value = round(original_value / scale)
```

Example: values 0–1000 mapped to 0–255 (unsigned int8):
- scale = 1000 / 255 = 3.92
- value 500 → round(500 / 3.92) = 128 ✓

### Symmetric vs Asymmetric quantization

- **Symmetric**: data centered around 0 (like batch-normalized weights). Zero-point = 0.
- **Asymmetric**: data skewed away from 0. Needs a zero-point offset.

### 1-bit LLM — BitNet 1.58 (Bonus, Krish Naik)

The next frontier: weights stored as only `-1`, `0`, or `1` (ternary).

- No matrix multiplication — only **addition** (anything × 1 = itself, anything × 0 = 0)
- Massive speedup: GPUs compute additions much faster than multiplications
- Research paper: BitNet b1.58 — matches FP16 accuracy from 3B parameters+
- Reduces latency, memory, energy consumption dramatically

---

## 4. LoRA — Low-Rank Adaptation

### Full form

**Lo**w-**R**ank **A**daptation of Large Language Models (Microsoft Research, 2021)

### The core idea (Krish Naik's intuition)

Instead of updating all weights of a model during fine-tuning, LoRA:
1. **Freezes** all original pre-trained weights (W₀)
2. **Injects** two small trainable matrices (A and B) alongside each frozen layer
3. Only trains A and B — which together approximate the weight update

### The math

The update to any weight matrix W is represented as:

```
W_new = W₀ + ΔW
ΔW = B × A        (matrix decomposition)
```

Where:
- W₀ is the frozen original weight matrix (e.g., shape 4096 × 4096)
- A is a small matrix of shape (r × d_in) — initialized randomly
- B is a small matrix of shape (d_out × r) — initialized to zero
- r = rank (a small number: 1, 2, 4, 8, 16, 64)

### Why this saves parameters — the math

Original W₀: 4096 × 4096 = **16.7 million** parameters to update

With LoRA (rank r=8):
- A: 8 × 4096 = 32,768
- B: 4096 × 8 = 32,768
- Total: **65,536** parameters → 256× fewer!

From Krish's table:

| Model size | Rank | Trainable params |
|---|---|---|
| 7B | 1 | 167K |
| 13B | 1 | 228K |
| 70B | 1 | 529K |
| 180B | 1 | 849K |
| 7B | 64 | ~86M |

Even at rank 64, you train 86M vs 7 billion = ~1.2% of parameters.

### Matrix rank — the intuition (Krish's chef analogy)

> "Rank = how many people are doing truly independent work. If 3 chefs all copy the same recipe — actual independent work = 1. Rank = 1."

Low rank means the weight update can be represented by a small amount of independent information — which is often true for domain adaptation tasks.

### Where LoRA is applied

LoRA is injected into the **transformer attention layers**:
- `q_proj` — Query projection
- `k_proj` — Key projection
- `v_proj` — Value projection
- `o_proj` — Output projection

Optionally also: `gate_proj`, `up_proj`, `down_proj` (MLP layers)

### When to use high rank?

> "If the model wants to learn **complex things** it wasn't trained for, use high rank. For simple domain adaptation, low rank (4–16) is enough." — Krish Naik

### LoRA scaling — alpha (α)

The LoRA output is scaled by `alpha / r` before being added to the frozen weights.

- Higher alpha → stronger LoRA updates relative to base model
- Common practice: set `lora_alpha = 2 × r` (e.g., r=16, alpha=32)

---

## 5. QLoRA — Quantized LoRA

### Full form

**Q**uantized **Lo**w-**R**ank **A**daptation (Dettmers et al., 2023 — University of Washington)

### What it adds on top of LoRA

QLoRA = LoRA + 4-bit quantization of the base model weights

```
Base LLM
   ↓
4-bit Quantization (freeze base weights in 4-bit)
   ↓
Frozen weights
   ↓
Inject LoRA adapters (FP16 precision)
   ↓
Train only adapters
   ↓
Save tiny adapter files
```

### Key innovations in QLoRA (from Patrik Szepesi course)

1. **4-bit NormalFloat (NF4)**: A new data type optimized for normally-distributed weights. Better than plain INT4 for neural network weights.

2. **Double Quantization**: Quantize the quantization constants themselves — saves ~0.4 bits per parameter extra.

3. **Paged Optimizers**: Uses NVIDIA unified memory to page optimizer states between GPU and CPU RAM. Avoids OOM errors during gradient spikes.

4. **Dequantization during compute**: Forward/backward pass dequantizes NF4 → FP16 on-the-fly, computes, then goes back to NF4 for storage.

### Memory savings comparison

| Technique | 7B model VRAM needed |
|---|---|
| Full FP32 | ~112 GB |
| Full FP16 | ~56 GB |
| LoRA (FP16 base) | ~28 GB |
| QLoRA (4-bit base) | ~6–10 GB |

**QLoRA makes fine-tuning a 7B model possible on a single 12GB consumer GPU.**

---

## 6. Key Hyperparameters Explained

### LoRA hyperparameters

| Param | What it controls | Typical values |
|---|---|---|
| `r` (rank) | Learning capacity. Higher = more expressive but more params | 4, 8, 16, 64 |
| `lora_alpha` | Strength of LoRA update. Scaled as alpha/r | Set to 2× rank |
| `lora_dropout` | Regularization — prevents overfitting of adapters | 0.05–0.1 |
| `target_modules` | Which layers get LoRA adapters | q_proj, k_proj, v_proj, o_proj |
| `bias` | Whether to train bias terms | "none" usually |
| `task_type` | Type of task | "CAUSAL_LM" for GPT-style |

### Quantization hyperparameters (BitsAndBytes)

| Param | What it controls |
|---|---|
| `load_in_4bit=True` | Quantize base model to 4-bit |
| `bnb_4bit_quant_type="nf4"` | Use NormalFloat4 (better than INT4 for weights) |
| `bnb_4bit_compute_dtype=torch.float16` | Compute in FP16 during forward/backward pass |
| `bnb_4bit_use_double_quant=True` | Apply double quantization for extra memory savings |
| `load_in_8bit=True` | Alternative: quantize to 8-bit (less aggressive) |

---

## 7. Full Pipeline: Step-by-Step Code

### Step 1 — Install packages

```python
# Patrik Szepesi course setup
%pip install accelerate peft bitsandbytes datasets transformers mlflow
```

### Step 2 — Import libraries

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import mlflow
from datasets import Dataset
```

### Step 3 — Choose your model

```python
# For 4-bit QLoRA (Mistral 7B — Patrik's course)
model_id = "mistralai/Mistral-7B-v0.1"

# For 8-bit (smaller model example)
# model_id = "facebook/opt-1.3b"
```

### Step 4 — Configure quantization

```python
# 4-bit QLoRA config (recommended)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for weights
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True      # Double quantization for extra savings
)

# OR: 8-bit config (less aggressive, less memory savings)
# bnb_config = BitsAndBytesConfig(load_in_8bit=True)
```

### Step 5 — Load the quantized model

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Required for causal LM

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically distributes across available GPUs
)

print("Model loaded successfully")
```

### Step 6 — Prepare model for kbit training

```python
# Required when using 4-bit or 8-bit — prepares model for gradient checkpointing
model = prepare_model_for_kbit_training(model)
```

### Step 7 — Configure LoRA adapters

```python
lora_config = LoraConfig(
    r=16,                    # Rank — controls learning capacity
    lora_alpha=32,           # Alpha — controls strength (= 2 × r is standard)
    target_modules=[         # Which attention layers get adapters
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,       # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"    # Causal language modeling (next-token prediction)
)

# Inject LoRA into the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output example: trainable params: 6,815,744 || all params: 3,752,071,168 || trainable%: 0.18%

print("QLoRA configured successfully")
```

### Step 8 — Prepare dataset

```python
# Example: instruction-following format
def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

# Tokenize
def tokenize(example):
    return tokenizer(
        format_prompt(example),
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# Apply to your dataset
tokenized_dataset = your_dataset.map(tokenize)
```

### Step 9 — Configure training arguments

```python
# TrainingArguments control: speed, memory, stability, checkpointing
training_args = TrainingArguments(
    output_dir="./mixtral-8x7b-qlora-dolly",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # Simulate larger batch size
    learning_rate=2e-4,
    fp16=False,
    bf16=True,                       # BFloat16 for better training stability
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    warmup_steps=10,
    optim="paged_adamw_8bit",        # 8-bit paged optimizer — saves memory
    report_to="mlflow",              # Log to MLflow
)

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # mlm=False because this is causal LM (next-token), not masked LM
)
```

### Step 10 — Train with MLflow tracking

```python
import time

run_name = f"mixtral-8x7b-qlora-{time.strftime('%Y%m%d-%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:
    # Log all hyperparameters
    mlflow.log_params({
        "model_id": model_id,
        "num_epochs": 2,
        "batch_size": 2,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "dataset_size": len(train_dataset)
    })
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print(f"Starting training... MLflow Run ID: {run.info.run_id}")
    
    # Train — only LoRA adapters are updated, base model stays frozen
    trainer.train()
    
    # Save adapter files (tiny — just MBs, not GBs)
    trainer.save_model("./final_model")
    
    # Register model in MLflow Model Registry
    mlflow.transformers.log_model(
        transformers_model={"model": trainer.model, "tokenizer": tokenizer},
        artifact_path="model",
        registered_model_name="mixtral-8x7b-qlora-dolly"
    )
    
    print("Training complete!")
    print("Model saved to: ./final_model")
```

---

## 8. Training Arguments Deep Dive

| Argument | Purpose | Guidance |
|---|---|---|
| `num_train_epochs` | How many full passes over data | Start with 1–3 |
| `per_device_train_batch_size` | Samples per GPU per step | Keep low (1–4) with 4-bit |
| `gradient_accumulation_steps` | Simulate larger batch without more memory | 4–8 |
| `learning_rate` | Step size for optimizer | 1e-4 to 3e-4 for QLoRA |
| `bf16=True` | Use BFloat16 for compute | Preferred on A100/H100 |
| `fp16=True` | Use Float16 for compute | Use on older GPUs (V100) |
| `warmup_steps` | Gradually increase LR at start | 10–100 |
| `save_steps` | Save checkpoint every N steps | 100–500 |
| `optim="paged_adamw_8bit"` | 8-bit AdamW with GPU→CPU paging | Always use with QLoRA |
| `gradient_checkpointing` | Recompute activations to save memory | Set True for very large models |

---

## 9. MLflow Tracking

MLflow tracks every training run — essential for enterprise governance.

```
MLflow tracks:
├── Hyperparameters (lr, rank, epochs, batch size)
├── Metrics (loss per step, eval loss)
├── Checkpoints (saved model snapshots)
└── Model Registry (versioned, deployable models)
```

Access via: `mlflow ui` → opens browser dashboard to compare all runs.

---

## 10. Pros and Cons

### LoRA

| Pros | Cons |
|---|---|
| 100–1000× fewer trainable parameters | Small loss in expressiveness vs full fine-tuning |
| Adapter files are tiny (MBs, not GBs) | Rank selection requires experimentation |
| Multiple task adapters on one base model | Not ideal for tasks requiring deep behavioral change |
| No inference latency (B×A merged at inference) | Target module selection can be tricky |
| Works with any transformer architecture | |

### QLoRA

| Pros | Cons |
|---|---|
| Fine-tune 7B model on 12GB GPU | 4-bit quantization introduces slight accuracy drop |
| ~4× memory reduction vs LoRA alone | Slower training than FP16 LoRA (dequant overhead) |
| NF4 preserves weight distribution better | BitsAndBytes library CUDA-only (no MPS/CPU) |
| Paged optimizers prevent OOM crashes | More complex setup than plain LoRA |
| Makes large model fine-tuning democratically accessible | Double quantization adds code complexity |

---

## 11. Alternatives to LoRA / QLoRA

| Technique | What it is | When to use |
|---|---|---|
| **Full Fine-Tuning** | Update all weights | When you have 8+ high-VRAM GPUs and need maximum accuracy |
| **Prefix Tuning** | Prepend trainable tokens to input | Simple task adaptation, very few parameters |
| **Adapter Layers** | Small bottleneck layers between transformer blocks | Pre-LoRA standard, more parameters than LoRA for same rank |
| **Prompt Tuning** | Only tune soft prompt embeddings | Extremely parameter-efficient, but limited capacity |
| **IA³** (Infused Adapter by Inhibiting and Amplifying Inner Activations) | Rescale activations with learned vectors | Even fewer params than LoRA, good for few-shot |
| **DoRA** (Weight-Decomposed LoRA) | Decomposes weights into magnitude + direction | More expressive than LoRA with similar cost |
| **LoftQ** | Quantize base + initialize LoRA better | Better accuracy than QLoRA at same memory |
| **GPTQ** | Post-training quantization (inference only) | Deployment, not fine-tuning |
| **GGUF / llama.cpp** | CPU-based quantized inference | Running models locally without GPU |
| **BitNet 1.58** | Ternary weights (-1, 0, 1) | Future: eliminates multiplication entirely |

---

## 12. Interview Cheat Sheet

**Q: What is LoRA?**
A: Low-Rank Adaptation. Freezes pre-trained weights, injects two small trainable matrices A and B per layer. ΔW = B × A. Trains only ~0.1–1% of parameters vs full fine-tuning.

**Q: What is QLoRA?**
A: Quantized LoRA. Loads base model in 4-bit NF4 format (saves ~4× memory vs FP16), then applies LoRA adapters in FP16. Enables fine-tuning a 7B model on a 12GB GPU.

**Q: What is rank in LoRA?**
A: Rank controls how many independent dimensions the weight update uses. Low rank (4–8) = fewer params, less capacity. High rank (64+) = more capacity for complex tasks. Rule: rank 8–16 works for most domain adaptation.

**Q: What is lora_alpha?**
A: Scaling factor. Applied as alpha/r to the LoRA output before adding to frozen weights. Higher alpha = stronger LoRA influence. Usually set to 2× rank.

**Q: What is NF4?**
A: NormalFloat4. A 4-bit data type that distributes its 16 quantization levels according to a normal distribution — which matches how neural network weights are naturally distributed. Better than plain INT4.

**Q: Why use paged_adamw_8bit optimizer?**
A: Stores optimizer states (momentum, variance) in 8-bit precision and pages them between GPU/CPU memory. Prevents OOM errors and saves GPU VRAM during training.

**Q: What is target_modules?**
A: The specific attention projections (q_proj, k_proj, v_proj, o_proj) where LoRA adapters are injected. These are the attention weight matrices in each transformer layer.

**Q: LoRA vs QLoRA — which to use?**
A: LoRA when you have enough VRAM (model fits in FP16). QLoRA when you're memory-constrained — 4-bit base + LoRA adapters cuts VRAM by ~4×.

**Q: How does QLoRA avoid accuracy loss from quantization?**
A: It uses Quantization-Aware Training (not post-training quantization), NF4 (which fits the weight distribution better), double quantization, and trains LoRA adapters in full FP16 precision even though the base is 4-bit.

**Q: What is gradient accumulation?**
A: Instead of one optimizer step per batch, accumulate gradients over N batches then step. Simulates a larger batch size without needing more VRAM.

---

*Guide compiled from: Krish Naik YouTube transcript (quantization + LoRA + QLoRA series), krishnaik06/Finetuning-LLM GitHub (Mistral/Mixtral QLoRA notebooks with MLflow), patrikszepesi/qlora-course GitHub (Fine_Tuning notebooks, deployment, Google Colab guides).*


3:27 PM
Looking at your guide, I'll extract all classes used in the QLoRA pipeline and explain each one's role and architecture.



All 10 classes/functions across 5 libraries — click any card to expand the full details. Here's the quick mental model:

transformers library is the backbone — BitsAndBytesConfig tells it how to quantize on load, AutoModelForCausalLM does the actual loading, AutoTokenizer handles text↔token conversion, TrainingArguments holds every training knob, DataCollatorForLanguageModeling assembles batches with causal labels, and Trainer runs the actual forward→loss→backward loop.

peft library is the LoRA layer — LoraConfig defines the adapter shape (rank, alpha, which layers), get_peft_model() injects those adapters and freezes everything else, and prepare_model_for_kbit_training() wires up gradient flow for quantized models (without it, backprop breaks on 4-bit weights).

datasets gives you Dataset — Arrow-backed tabular storage for your instruction-output pairs, with efficient map() for tokenization.

mlflow wraps the whole run — params in, model artifact out, versioned in the Model Registry.

torch contributes the dtype choices (float16, bfloat16) that determine compute precision during training — distinct from storage precision (NF4).

ask me questions basic to tough to test my understanding and eavlauate and give correct answer

4:23 PM


Loading questions basic to tough...



