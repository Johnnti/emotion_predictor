**Emotion Predictor**

**Overview**
- **Purpose:** Train a lightweight transformer (DistilBERT) to predict emotions from short text inputs. The repository contains training code, model artifacts under `distilbert-emotion-model/`, and checkpoints.
- **Contents:** `train_emotion.py` training script, `distilbert-emotion-model/` (weights & checkpoints), and utility/tokenizer files.
## Emotion Predictor

This project fine-tunes DistilBERT to classify emotions from text.

### Features
- Train your own emotion classifier using `train_emotion.py`.
- Pretrained model and tokenizer files in `distilbert-emotion-model/`.
- Easily load and use the model locally or from Hugging Face Hub.

### Quick Start

**1. Install dependencies:**
```bash
pip install torch transformers datasets
```

**2. Train the model:**
```bash
python train_emotion.py --train_file data/train.jsonl --validation_file data/valid.jsonl --output_dir distilbert-emotion-model
```

**3. Run inference locally:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('distilbert-emotion-model')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-emotion-model')
text = "I am happy!"
inputs = tokenizer(text, return_tensors='pt')
pred = model(**inputs).logits.argmax(-1).item()
print(pred)
```

**4. Load from Hugging Face Hub:**
Replace `your-username/emotion-predictor-distilbert` with your model repo name.
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('your-username/emotion-predictor-distilbert')
model = AutoModelForSequenceClassification.from_pretrained('your-username/emotion-predictor-distilbert')
```

### Push your model to Hugging Face
```python
model.push_to_hub('your-username/emotion-predictor-distilbert')
tokenizer.push_to_hub('your-username/emotion-predictor-distilbert')
```

### Notes
- Add `distilbert-emotion-model/` to your `.gitignore` to avoid pushing large files to GitHub.
- For a web demo, create a Hugging Face Space with Gradio and load your model from the Hub.

---
**MIT License**

**Technique & Model**
- **Model architecture:** DistilBERT (a smaller, faster distilled version of BERT) fine-tuned for multi-class emotion classification using a classification head on top of pooled token embeddings.
- **Loss & optimization:** Cross-entropy loss for multi-class classification; typically AdamW optimizer with weight decay and a learning-rate scheduler. During training we saved checkpoints under `distilbert-emotion-model/checkpoint-*`.
- **Training strategy:** Standard fine-tuning of pre-trained transformer weights with a small learning rate (e.g., 2e-5), gradual warmup, and early stopping/evaluation on a validation split. Use mixed precision (fp16) and gradient accumulation for larger effective batch sizes if needed.

**Dataset**
- **What to expect:** The training pipeline expects a dataset of (text, label) pairs. Labels are integer-coded emotion classes. The original training dataset used for this repo should be documented or referenced here if available; if you used a public dataset (for example, a subset of the GoEmotions or a custom-labeled dataset), include the citation and license.
- **Preprocessing:** Basic text normalization, tokenization using the saved tokenizer files in `distilbert-emotion-model/` (`tokenizer.json`, `vocab.txt`, etc.), optional lowercasing/truncation to `max_length` (commonly 128 or 256 tokens).

**Reproduce Training Locally**
Prerequisites
- Python 3.8+ and a GPU (recommended).
- Install dependencies (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# If no requirements.txt, at minimum:
pip install transformers datasets accelerate torch huggingface_hub safetensors
```

Run training

```bash
# Basic command (example - adjust flags inside `train_emotion.py` as needed):
python3 train_emotion.py \
	--train_file data/train.jsonl \
	--validation_file data/valid.jsonl \
	--output_dir distilbert-emotion-model \
	--model_name_or_path distilbert-base-uncased \
	--per_device_train_batch_size 16 \
	--learning_rate 2e-5 \
	--num_train_epochs 3
```

Notes
- The repository's `train_emotion.py` should contain argument parsing for dataset paths, model selection, and Trainer/Accelerate settings. If you need help adapting flags, open `train_emotion.py` and I can add example CLI arguments.

**Evaluate & Test Locally**
- Quick inference example using the saved model (Python):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('distilbert-emotion-model')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-emotion-model')

def predict(text):
		inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
		with torch.no_grad():
				logits = model(**inputs).logits
		probs = torch.nn.functional.softmax(logits, dim=-1)
		pred = torch.argmax(probs, dim=-1).item()
		return pred, probs[0].tolist()

print(predict("I am thrilled with this progress!"))
```

**Push Model to Hugging Face Hub (recommended)**
- Using the `huggingface_hub` and `transformers` utilities you can host model weights externally and keep your Git repo lightweight.

Option A — `transformers` `push_to_hub` (inside a training script or notebook):

```python
from transformers import AutoModelForSequenceClassification
model.push_to_hub("your-username/emotion-predictor-distilbert")
tokenizer.push_to_hub("your-username/emotion-predictor-distilbert")
```

Option B — Hugging Face CLI

```bash
pip install huggingface_hub
hf login  # authenticate
# From inside the `distilbert-emotion-model/` folder or using the model card API
transformers-cli upload ./distilbert-emotion-model --repo_id your-username/emotion-predictor-distilbert
```

After upload, the model will be available on Hugging Face and you can create a Space (Streamlit/Gradio) to serve a UI.

**Deploying a Simple UI on Hugging Face Spaces**
- You can create a `Gradio` app that loads the model from the Hub and provides a small web UI.
- Minimal `app.py` sketch:

```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('your-username/emotion-predictor-distilbert')
model = AutoModelForSequenceClassification.from_pretrained('your-username/emotion-predictor-distilbert')

def infer(text):
		inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
		with torch.no_grad():
				logits = model(**inputs).logits
		probs = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
		return {str(i): p for i, p in enumerate(probs)}

iface = gr.Interface(fn=infer, inputs="text", outputs="label")
iface.launch()
```

UI version notes
- Use `gradio` (recommended for quick UI) or `streamlit` for a simple web demo.
- On Spaces, select an instance type that fits model size (CPU for small models; GPU for faster inference / larger models).

**Using the model from Hugging Face (load remotely)**

- If the model is hosted on Hugging Face Hub (for example `your-username/emotion-predictor-distilbert`), you can load it directly without downloading files manually. This is the simplest way for users to run inference.

Local Python example (public model):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_ID = "your-username/emotion-predictor-distilbert"  # replace with the real repo id
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

def predict(text: str):
	inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
	with torch.no_grad():
		logits = model(**inputs).logits
	probs = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
	return probs

print(predict("I am happy today"))
```

- Private models: run `huggingface-cli login` (or set `HF_HOME` and `HUGGINGFACE_HUB_TOKEN` environment variable) so `transformers` can authenticate and download private files.

```bash
pip install huggingface_hub
huggingface-cli login
# or set token manually
export HUGGINGFACE_HUB_TOKEN="hf_xxx..."
```

**Push your own model to Hugging Face (recommended workflows)**

There are two convenient ways to push a model you trained locally to the Hub so others can load it directly.

1) Quick push from a Python script (recommended during/after training)

```python
# after training and saving a local checkpoint
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# assume `model` and `tokenizer` are in memory (or loaded from local dir)
model.push_to_hub("your-username/emotion-predictor-distilbert")
tokenizer.push_to_hub("your-username/emotion-predictor-distilbert")
```

2) Repository workflow (git + Git LFS) — good for reproducibility and model cards

```bash
pip install git-lfs huggingface_hub
git lfs install
# create a new model repo on the Hub (public or private)
huggingface-cli repo create emotion-predictor-distilbert --type model --private

# clone the new repo, copy files, commit and push
git clone https://huggingface.co/your-username/emotion-predictor-distilbert
cd emotion-predictor-distilbert
# copy your files (pytorch model, safetensors, config, tokenizer files)
cp -r ../distilbert-emotion-model/* ./
git add .
git commit -m "Initial upload of DistilBERT emotion model"
git push
```

Alternative: use `huggingface_hub.Repository` from Python to push without manual git commands:

```python
from huggingface_hub import Repository
repo = Repository("./hf-model", clone_from="your-username/emotion-predictor-distilbert")
model.save_pretrained(repo.local_dir)
tokenizer.save_pretrained(repo.local_dir)
repo.push_to_hub(commit_message="Upload model")
```

Model card
- Add a `README.md` to the model repository (this is the model card shown on Hugging Face). Include: model description, training dataset citation & license, intended use, evaluation metrics, limitations, and a license.

**Creating a Space (Gradio) to demo the model**

- Create a new Space at https://huggingface.co/spaces and choose `Gradio` or `Streamlit`.
- Add an `app.py` similar to the earlier example and a `requirements.txt` listing `transformers`, `torch`, `gradio`, and `huggingface_hub`.

Example `requirements.txt` for a Gradio Space:

```
transformers
torch
gradio
huggingface_hub
safetensors
```

**Helpful tips & gotchas**
- Use `safetensors` for saving model weights for faster, safer uploads when supported.
- If your repo is private, ensure the Space or runner has permission to read the model (or mark model public).
- Keep a human-readable `model card` explaining dataset sources and appropriate/unsafe uses.
- If weights are large, prefer Git LFS (Hub uses LFS under the hood for large files) or use the `push_to_hub` which handles large files automatically.

---

*Appendix:* quick checklist for sharing your model on HF

- [ ] Create HF account and login locally with `huggingface-cli login`.
- [ ] Create a model repo using `huggingface-cli repo create` or via the web UI.
- [ ] Save model/tokenizer with `save_pretrained` and push with `push_to_hub` or git + LFS.
- [ ] Add a model card (`README.md`) with dataset & license.
- [ ] (Optional) Create a Space with `app.py` and `requirements.txt` to demo the model.


