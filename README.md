# Pixels to Predictions — Reproduction Guide

all adapters- https://drive.google.com/drive/folders/1ievuBdXvOD0nHxe9hnSw3VYRtXKQm0US?usp=sharing

Fine-tuning [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
on multimodal science multiple-choice QA, achieving **0.93360** on the public leaderboard
via a weighted ensemble of 27 checkpoints across six training phases.

These instructions target **Google Colab with an A100 GPU** (40 GB VRAM).
The full pipeline was developed on an NVIDIA GeForce RTX 3090 (24 GB) under WSL2 Ubuntu 24
but all numbers quoted below are from Colab unless noted otherwise.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Key Configuration Constants](#key-configuration-constants)
5. [Critical Image Processor Settings](#critical-image-processor-settings)
6. [Quantization Configuration](#quantization-configuration)
7. [Generating a Submission from a Checkpoint](#generating-a-submission-from-a-checkpoint)
8. [Running the Notebooks in Order](#running-the-notebooks-in-order)
9. [Reproducing the Best Single Model](#reproducing-the-best-single-model-connector-epoch-7)
10. [Reproducing the Best Ensemble](#reproducing-the-best-ensemble-result-093360)
11. [Session Persistence and Resuming Training](#session-persistence-and-resuming-training)
12. [Common Errors and Fixes](#common-errors-and-fixes)
13. [Environment Versions](#environment-versions-development)

---

## Hardware Requirements

| Resource | Minimum | Used in this study |
|----------|---------|-------------------|
| GPU | A100 40GB | RTX 3090 24GB |
| System RAM | 25 GB | 15 GB |
| Disk (checkpoints) | 10 GB | ~8 GB on Google Drive |
| Colab tier | Pro (A100 access) | N/A (local machine) |

The T4 (16 GB) on free Colab is insufficient. The V100 (16 GB) on Colab Pro may work
at `BATCH_SIZE=1` but is not recommended. An A100 (40 GB) provides comfortable headroom
for all configurations including the connector run (4.79M trainable parameters).

---

## Repository Structure

```
pixels/
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── images/
│       ├── train/
│       ├── val/
│       └── test/
├── logs/                              # created automatically during training
├── submission/                        # output CSVs
├── smolvlm_finetune_v4.ipynb          # log-likelihood runs (Phases 3-4)
├── nli_training.ipynb                 # NLI fine-tuning (Phase 5)
├── dora_nri.ipynb                     # DoRA + Vision LoRA (Phase 6)
├── cross_attention_dora_nli.ipynb     # Connector LoRA (Phase 6)
├── dora_NLI_minimax.ipynb             # Minimax training (Phase 8)
├── dora_dual_pipeline.ipynb           # Ensemble generation and submission
├── test_accuracy.ipynb                # Train-set accuracy evaluation
└── README.md
```

> **Images are not included** in this repository due to size. Download the dataset
> from the Kaggle competition page and place the `images/` folder under `data/`.

---

## Environment Setup

### Step 1 — Mount Google Drive

Run this at the top of every notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create this structure on your Drive before running anything:

```
MyDrive/
└── pixels/
    ├── data/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── images/
    └── logs/       # created automatically
```

### Step 2 — Install Dependencies

```python
!pip install \
    transformers==4.47.0 \
    peft==0.13.2 \
    bitsandbytes==0.44.1 \
    accelerate==1.2.0 \
    pillow \
    tqdm \
    pandas \
    matplotlib \
    psutil \
    --quiet
```

> **Version pinning matters.** The bitsandbytes CUDA backend changed significantly
> between releases. Version 0.44.1 was used throughout development. A newer version
> may trigger the `_check_is_size` FutureWarning or alter PagedAdamW8bit behaviour.

### Step 3 — Verify GPU

```python
import torch

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU:  {gpu_name}')
print(f'VRAM: {vram_gb:.1f} GB')
assert vram_gb >= 35, f'Insufficient VRAM ({vram_gb:.1f} GB). Need an A100.'
print('GPU check passed.')
```

### Step 4 — Update Data and Log Paths

```python
from pathlib import Path

DATA_DIR = Path('/content/drive/MyDrive/pixels/data')
LOG_DIR  = Path('/content/drive/MyDrive/pixels/logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)

assert (DATA_DIR / 'train.csv').exists()
assert (DATA_DIR / 'val.csv').exists()
assert (DATA_DIR / 'test.csv').exists()
assert (DATA_DIR / 'images').exists()
print('Data paths verified.')
```

### Step 5 — Adjust Batch Size

On an A100 the default configuration works without modification:

```python
BATCH_SIZE = 4
GRAD_ACCUM = 4   # effective batch size = 16
```

If you get OOM errors (unlikely on A100), reduce to:

```python
BATCH_SIZE = 2
GRAD_ACCUM = 8   # effective batch size still = 16
```

---

## Key Configuration Constants

These appear at the top of each notebook. Set them before running any training cell.

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `HuggingFaceTB/SmolVLM-500M-Instruct` | Base model HuggingFace ID |
| `DATA_DIR` | `Path('data')` | Path to dataset root |
| `LOG_DIR` | `Path('logs')` | Where checkpoints and logs are saved |
| `BATCH_SIZE` | `4` | Per-device training batch size |
| `GRAD_ACCUM` | `4` | Gradient accumulation steps |
| `LR` | `2e-4` | Learning rate |
| `EPOCHS` | `5` or `10` | Number of training epochs |
| `LORA_R` | `16` | LoRA rank — higher rank = more capacity, more VRAM |
| `LORA_ALPHA` | `32` | LoRA scaling factor (conventionally 2x rank) |
| `MAX_CONTEXT_CHARS` | `2000` | Truncation limit for lecture/hint text fields |
| `IMG_SIZE` | `512` | Image resolution fed to the processor |
| `SEED` | `42` | Global random seed for reproducibility |

---

## Critical Image Processor Settings

These three lines must appear before any training or inference. Omit them and each
sample will consume ~1,352 tokens instead of ~292, making training intractable:

```python
processor.image_processor.do_image_splitting = False
processor.image_processor.size               = {"longest_edge": 512}
processor.image_processor.max_image_size     = {"longest_edge": 512}
```

SmolVLM's default processor tiles images into sub-patches, generating over 1,100
visual tokens per image. Disabling splitting and capping resolution at 512 px
cuts this to roughly 292 tokens and reduces per-epoch training time from ~13 hours
to ~8 minutes on the full 3,109-sample training set.

---

## Quantization Configuration

All notebooks load the base model in 4-bit NF4:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

The base model occupies approximately 0.94 GB in 4-bit. Do not load it in 8-bit
or 16-bit — VRAM usage increases 2-4x and training will OOM even on an A100.

All notebooks use `PagedAdamW8bit` as the optimizer:

```python
from bitsandbytes.optim import PagedAdamW8bit
optimizer = PagedAdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
```

Standard AdamW stores full-precision momentum buffers and will OOM at ~4-5M
trainable parameters. PagedAdamW8bit pages optimizer states to GPU in 8-bit,
reducing the memory footprint by approximately 75%.

---

## Generating a Submission from a Checkpoint

This is the most common task after training: load a saved LoRA adapter and
run inference on the test set to produce a `submission.csv`.

### What a checkpoint looks like

Each epoch saves a folder under `logs/`:

```
logs/
└── connector_ckpt_epoch7_20260501_130329/
    ├── adapter_config.json        # LoRA architecture (rank, alpha, target modules)
    ├── adapter_model.safetensors  # trained LoRA weights
    └── README.md
```

The `adapter_model.safetensors` file contains the trained LoRA weights.
The `adapter_config.json` records the exact LoRA configuration so `PeftModel`
can reconstruct the layers without any manual configuration.

### Minimal inference script

Copy this into a notebook cell or standalone script. Change only `CHECKPOINT`:

```python
import ast
import torch
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "HuggingFaceTB/SmolVLM-500M-Instruct"
CHECKPOINT    = Path("logs/connector_ckpt_epoch7_20260501_130329")  # change this
DATA_DIR      = Path("data")
OUTPUT_CSV    = Path("submission/my_submission.csv")
MAX_CTX_CHARS = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load processor ────────────────────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(MODEL_ID)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# CRITICAL — must be set before any processor() call
processor.image_processor.do_image_splitting = False
processor.image_processor.size               = {"longest_edge": 512}
processor.image_processor.max_image_size     = {"longest_edge": 512}

# ── Load base model in 4-bit ──────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
base_model.config.use_cache = False

# ── Attach LoRA adapter ───────────────────────────────────────────────────────
model = PeftModel.from_pretrained(base_model, CHECKPOINT)
model.eval()
print(f"Loaded adapter from {CHECKPOINT}")

# ── Load test data ────────────────────────────────────────────────────────────
test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df["choices"] = test_df["choices"].apply(ast.literal_eval)

# ── NLI Yes/No scoring ────────────────────────────────────────────────────────
# For each answer choice, ask the model "Is [choice] correct? Yes or No?"
# The margin log P(Yes) - log P(No) is the confidence score; argmax wins.

YES_ID = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO_ID  = processor.tokenizer.encode("No",  add_special_tokens=False)[0]

def build_nli_prompt(row, choice, max_context_chars=2000):
    text = ""
    if pd.notna(row.get("lecture")) and row["lecture"]:
        text += f"Background:\n{row['lecture'][:max_context_chars]}\n\n"
    if pd.notna(row.get("hint")) and row["hint"]:
        text += f"Passage:\n{row['hint'][:max_context_chars]}\n\n"
    text += f"Question: {row['question']}\n"
    text += f"Proposed answer: {choice}\n"
    text += "Is this the correct answer? Answer Yes or No."
    return f"<|im_start|>User:<image>{text}<end_of_utterance>\nAssistant:"

def predict_nli(model, processor, row, image):
    margins = []
    for choice in row["choices"]:
        prompt = build_nli_prompt(row, choice, MAX_CTX_CHARS)
        inputs = processor(
            text=[prompt], images=[image],
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        margin = (log_probs[0, YES_ID] - log_probs[0, NO_ID]).item()
        margins.append(margin)
        torch.cuda.empty_cache()
    return int(torch.tensor(margins).argmax().item())

# ── Run inference on test set ─────────────────────────────────────────────────
predictions = []
for _, row in test_df.iterrows():
    image = Image.open(DATA_DIR / row["image_path"]).convert("RGB")
    pred  = predict_nli(model, processor, row, image)
    predictions.append({"id": row["id"], "answer": pred})

# ── Write submission CSV ──────────────────────────────────────────────────────
OUTPUT_CSV.parent.mkdir(exist_ok=True)
pd.DataFrame(predictions).to_csv(OUTPUT_CSV, index=False)
print(f"Submission written to {OUTPUT_CSV}  ({len(predictions)} rows)")
```

### Choosing which checkpoint to use

The checkpoint folder name encodes the run name, epoch, and timestamp:

```
logs/connector_ckpt_epoch7_20260501_130329/
      ^              ^      ^
      run name       epoch  run timestamp (YYYYMMDD_HHMMSS)
```

Guidelines for picking the best epoch:

| Run | Best epoch | Val accuracy | LB score |
|-----|-----------|-------------|---------|
| NLI | 4 or 5 | ~0.898 | ~0.897 |
| NLI resumed | 5 | ~0.908 | ~0.899 |
| DoRA + Vision | 9 | ~0.896 | ~0.907 |
| Connector | **7** | ~0.910 | **~0.920** |
| Minimax | 11 or 12 | varies | varies |

When in doubt, pick the checkpoint with the highest val accuracy logged during
training. The JSON training logs in `logs/` contain per-epoch val accuracy for
every run.

### Reading val accuracy from a log file

```python
import json

with open("logs/connector_log_20260501_130329.json") as f:
    log = json.load(f)

for entry in log:
    print(f"Epoch {entry['epoch']:2d}  val_acc={entry['val_acc']:.4f}")
```

### Swapping to a different adapter

Change only the `CHECKPOINT` path — everything else stays the same:

```python
# DoRA + Vision, epoch 9
CHECKPOINT = Path("logs/dora_vision_resume_epoch9_20260430_175706")

# NLI epoch 4
CHECKPOINT = Path("logs/nli_ckpt_epoch4_20260429_101700")
```

`PeftModel` reads `adapter_config.json` inside the folder and automatically
reconstructs the correct LoRA layers. No other configuration changes are needed.

### Running the full weighted ensemble

Open `dora_dual_pipeline.ipynb`. The ensemble cell lists all 27 checkpoint paths
with their val-accuracy weights. For each checkpoint it runs `predict_nli` on
every test sample and accumulates the weighted Yes/No margins. The final prediction
is the argmax of the accumulated weighted margins across all checkpoints. This takes
approximately 4-5 hours on an A100 due to the sequential checkpoint loading.

---

## Running the Notebooks in Order

The notebooks are designed to be run sequentially. Each saves all outputs to
`LOG_DIR` and the next notebook loads from there.

### Phase 3-4: Log-Likelihood Training

```
smolvlm_finetune_v4.ipynb
```

Initial log-likelihood training (20+ epochs) plus VRAM, module, and LR ablation
studies. Run this first to establish the baseline and understand the training dynamics.

### Phase 5: NLI Fine-Tuning

```
nli_training.ipynb
```

Switches the training objective from log-likelihood to Yes/No NLI scoring. Run for
5 epochs, then optionally resume for 2 more using the resume cell. This is the first
checkpoint worth submitting (expected ~0.897 LB).

### Phase 6: Architecture Refinement

```
dora_nri.ipynb                  (DoRA + Vision LoRA, 10 epochs)
cross_attention_dora_nli.ipynb  (adds Connector LoRA, 10 epochs)
```

Run `dora_nri.ipynb` first. The connector notebook then initialises from the best
DoRA checkpoint. The connector epoch 7 checkpoint is the strongest single model.

### Phase 8: Advanced Training

```
dora_NLI_minimax.ipynb
```

Adversarial minimax training: 3 pretrain epochs followed by 12 minimax epochs where
the model is trained against dynamically reweighted hard examples.

### Evaluation and Submission

```
test_accuracy.ipynb         (full train-set accuracy sanity check)
dora_dual_pipeline.ipynb    (weighted ensemble → final submission CSV)
```

---

## Reproducing the Best Single Model (Connector Epoch 7)

To get a competitive result without running the full pipeline:

1. Run `nli_training.ipynb` for 5 epochs — approximately 40 minutes on A100.
2. Run `cross_attention_dora_nli.ipynb` for exactly 7 epochs — approximately 55 minutes.
3. Use the inference script above with `CHECKPOINT = Path("logs/connector_ckpt_epoch7_{run_id}")`.

**Expected result:** ~0.9195 on the public leaderboard.

---

## Reproducing the Best Ensemble Result (0.93360)

All training runs must be completed first. Estimated wall-clock time on an A100:

| Run | Epochs | Estimated Time |
|-----|--------|---------------|
| NLI training | 7 | ~55 min |
| DoRA + Vision | 10 | ~80 min |
| Connector | 10 | ~80 min |
| Minimax | 15 | ~120 min |
| **Total** | **42** | **~5.5 hours** |

Then open `dora_dual_pipeline.ipynb` and run the ensemble cell.

---

## Session Persistence and Resuming Training

Colab sessions disconnect after inactivity (90 minutes on free tier, up to 24 hours
on Pro+). All training notebooks include resume cells. To resume after disconnection:

1. Rerun all setup cells (mount Drive, install dependencies, load processor, build dataset).
2. In the resume cell, set `RESUME_EPOCH` to the last epoch that finished:

```python
RESUME_EPOCH = 3   # last epoch that finished saving to Drive
```

3. The resume cell loads the checkpoint, reinitialises the optimizer and scheduler,
   and continues from the next epoch.

> Checkpoints are saved at the end of every epoch. If a session dies mid-epoch,
> that epoch's progress is lost and training restarts from the last complete epoch.

### Keep-Alive (optional)

Paste this in the browser console (F12) to prevent session timeout during long runs:

```javascript
function KeepAlive() {
  document.querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect').click();
}
setInterval(KeepAlive, 60000);
```

---

## Common Errors and Fixes

### `CUDA out of memory`
1. Reduce `BATCH_SIZE` to 2 and increase `GRAD_ACCUM` to 8.
2. Confirm `processor.image_processor.do_image_splitting = False` is set.
3. Confirm `model.config.use_cache = False` is set before training.
4. Add `torch.cuda.empty_cache()` after each inference choice loop.

### `AssertionError: CUDA not available`
PyTorch was installed without CUDA support:
```python
!pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
```
Then restart the runtime.

### `FutureWarning: _check_is_size will be removed`
Cosmetic warning from bitsandbytes >= 0.43.0. Safe to ignore.

### `RuntimeError: Expected all tensors to be on the same device`
Occurs in the minimax loop when the auxiliary MLP is on CPU. Fix:
```python
aux_loss = -(norm_w2 * per_sample_loss2.detach().cpu()).sum()
```

### `FileNotFoundError: train.csv not found`
Drive not mounted or path wrong:
```python
import os
print(os.listdir('/content/drive/MyDrive/pixels/data'))
```

### `Checkpoint not found` when resuming
List what is actually saved on Drive:
```python
import os
print(os.listdir('/content/drive/MyDrive/pixels/logs'))
```
Set `RESUME_EPOCH` to the last epoch number visible in that listing.

### Inference produces the same answer for every sample
The image processor settings were not applied before the first `processor()` call.
Restart the runtime, rerun all setup cells, and ensure `do_image_splitting = False`
and `size = {"longest_edge": 512}` are set before loading the model.

---

## Environment Versions (Development)

| Library | Version |
|---------|---------|
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu128 |
| transformers | 4.47.0 |
| peft | 0.13.2 |
| bitsandbytes | 0.44.1 |
| accelerate | 1.2.0 |
| pillow | 10.4.0 |
| pandas | 2.2.3 |
| matplotlib | 3.9.2 |
| numpy | 1.26.4 |

To install the exact development environment:

```python
!pip install \
    transformers==4.47.0 \
    peft==0.13.2 \
    bitsandbytes==0.44.1 \
    accelerate==1.2.0 \
    pillow==10.4.0 \
    pandas==2.2.3 \
    matplotlib==3.9.2 \
    numpy==1.26.4 \
    --quiet
```

PyTorch is pre-installed on Colab with CUDA support and does not need to be reinstalled
unless the Colab version differs significantly from 2.5.1.

---
