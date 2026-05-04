# QuantumChem-200K: A Large-Scale Open Organic Molecular Dataset for Quantum-Chemistry Property Screening and Language Model Benchmarking

We introduce QuantumChem-200K, a large-scale dataset of over 200,000 organic molecules annotated with seven quantum-chemical properties, including two-photon absorption (TPA) cross sections, TPA spectral ranges, singlet–triplet intersystem crossing (ISC) energies, toxicity and synthetic accessibility scores, solubility, and boiling point. These values are computed using a hybrid workflow that integrates density function theory (DFT), semi-empirical excited-state methods, atomistic quantum solvers, and neural-network predictors.
Data is available at: https://huggingface.co/datasets/YinqiZeng704/200k_monomer_properties, and paper is avalibale at: https://arxiv.org/abs/2511.21747

# Dataset composition:
<img width="658" height="230" alt="302ddf4f-5268-42f3-b504-65543633cdbc" src="https://github.com/user-attachments/assets/6b459694-6491-42ad-9457-f549e1ee091d" />
<img width="483" height="144" alt="fdbe4b5e-8b89-45c5-b0cb-00caea4d5c22" src="https://github.com/user-attachments/assets/8c86162b-9885-41de-86e4-a8f50d1dc488" />

# Data curation process:
<img width="900" height="700" alt="Screenshot 2025-11-30 at 18 25 57" src="https://github.com/user-attachments/assets/7264ac7e-b3bf-4b61-a51e-7ee792d0e64d" />

# LLM-Based Monomer Property Prediction

This repository contains scripts and notebooks for fine-tuning and evaluating large language models for **SMILES-to-monomer property prediction**. Given a molecular SMILES string, the model predicts relevant photochemical, physical, and synthetic properties such as:

- sigma at 780 nm
- maximum sigma
- ISC energy
- toxicity score
- synthetic accessibility score
- boiling point
- solubility

The main workflow includes:

1. Fine-tuning a Qwen2.5-32B model with LoRA using Unsloth.
2. Running batch inference on SMILES strings.
3. Comparing model predictions against ground-truth property tables using weighted MAE, RMSE, Pearson r. 
4. Running Claude Haiku, Deepseek, Gemma, Llama, Phi, Mistral, GPT 5.2, and graph learning model baselines for comparison, using wMAE, precision, recall, Pareto precision, and hypervolume regret.

---

## Repository Structure

```text
.
├── fine_tuning.py              # LoRA fine-tuning script using Unsloth + Qwen2.5-32B
├── infer.ipynb                 # Batch inference notebook using a fine-tuned LoRA adapter
├── claude_infer.py             # Claude baseline inference script
├── wmae_eval_sqrt.ipynb        # Weighted MAE evaluation notebook with sqrt rebalancing
├── requirements.txt            # Python dependencies
├── 100testbank.csv            # Input SMILES file for batch inference
├── 3000testbank.csv             # Ground-truth test set for evaluation
├── 100prediction.csv           # Prediction CSV for evaluation
└── outputs/                    # Training checkpoints and model outputs
```

---

## Setup

**Requirements:** Python 3.10 or newer is recommended. A CUDA-capable NVIDIA GPU is strongly recommended because the training and inference scripts load Qwen2.5-32B in 4-bit mode and move tensors to CUDA.

### 1. Create and activate a virtual environment

```bash
cd /path/to/your/repo
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

A minimal `requirements.txt` may include:

```text
torch
transformers
datasets
trl
peft
accelerate
bitsandbytes
unsloth
pandas
numpy
matplotlib
ipython
jupyter
anthropic
```

For GPU-accelerated PyTorch, install the CUDA build that matches your system from the official PyTorch installation page:

```text
https://pytorch.org/get-started/locally/
```

---

## Environment Variables

Before running inference or uploading models, set your API tokens as environment variables.

```bash
export HF_TOKEN="your_huggingface_token"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

Do **not** hard-code tokens in notebooks or scripts before pushing to GitHub.

In `infer.ipynb`, replace any hard-coded Hugging Face token with:

```python
import os
HF_TOKEN = os.getenv("HF_TOKEN")
```

In `claude_infer.py`, replace:

```python
api_key=os.getenv("your api key")
```

with:

```python
api_key=os.getenv("ANTHROPIC_API_KEY")
```

---

## 1. Fine-Tuning the Model

The main training script is:

```bash
python fine_tuning.py
```

This script fine-tunes:

```text
Base model: unsloth/Qwen2.5-32B
Dataset: YinqiZeng704/200k_monomer_properties
Method: LoRA fine-tuning
Quantization: 4-bit
Output directory: outputs/
```

The training data are formatted using an Alpaca-style prompt:

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Based on the given SMILES string, predict the monomer's relevant properties.

### Input:
<SMILES>

### Response:
<property prediction>
```

### Important checkpoint note

The current script resumes training from a hard-coded checkpoint path:

```python
trainer.train(resume_from_checkpoint="/mnt/shared/gpfs/home/renjie2/fine_tune/forward/outputs/checkpoint-3000")
```

For a fresh training run, change this to:

```python
trainer.train()
```

Or replace the checkpoint path with your own local checkpoint path.

### Training outputs

The fine-tuning script produces:

```text
outputs/              # model checkpoints
training_log.csv      # exported training log
```

The script can also push the trained adapter and tokenizer to Hugging Face:

```python
model.push_to_hub("YinqiZeng704/200k_model", token="your token")
tokenizer.push_to_hub("YinqiZeng704/200k_model", token="your token")
```

Before public release, replace the token with `HF_TOKEN` from the environment.

---

## 2. Running Inference with the Fine-Tuned Model

Open the inference notebook:

```bash
jupyter notebook infer.ipynb
```

or:

```bash
jupyter lab infer.ipynb
```

The notebook loads:

```text
Base model: unsloth/Qwen2.5-32B
LoRA adapter: YinqiZeng704/200k_model_v2
```

It runs a single test prediction and then performs batch inference on:

```text
3000bestbank.csv
```

The CSV should contain a column named:

```text
SMILES
```

Example:

```csv
SMILES
CC12NC1C1C(C#N)C21
C=C(C)OC
NC(=O)C1=CCCCC1
```

The notebook writes batch outputs to:

```text
3000outputs.txt
```

---

## 3. Running Baseline Inference

E.g., the Claude baseline script is:

```bash
python claude_infer.py
```

This script performs:

1. A single test prediction for one SMILES string.
2. Batch inference over the first 3000 SMILES strings in `3000.csv`.
3. Output saving to `3000outputs_claude_haiku.txt`.

Expected input:

```text
3000testbank.csv
```

Required column:

```text
SMILES
```

Output file:

```text
3000outputs_claude_haiku.txt
```

The script currently uses:

```text
model = claude-haiku-4.5
temperature = 0.2
max_tokens = 256
```

---

## 4. Evaluating Predictions with Weighted MAE

Open the evaluation notebook:

```bash
jupyter notebook wmae_eval_sqrt.ipynb
```

or:

```bash
jupyter lab wmae_eval_sqrt.ipynb
```

The notebook compares a ground-truth CSV and a prediction CSV:

```python
truth_path = "100testbank.csv"
pred_path  = "100prediction.csv"
```

It excludes:

```python
exclude = {"wavelength_range"}
```

The notebook computes weighted mean absolute error using square-root rebalancing:

```math
w_i =
\left(\frac{1}{r_i}\right)
\left(
\frac{K\sqrt{1/n_i}}
{\sum_{j=1}^{K}\sqrt{1/n_j}}
\right)
```

```math
\mathrm{wMAE}
=
\frac{1}{|M|}
\sum_{i=1}^{K}
w_i
\sum_{m \in \mathrm{valid}_i}
|y_i(m)-\hat{y}_i(m)|
```

Where:

- `K` is the number of evaluated properties.
- `M` is the number of evaluated molecules.
- `n_i` is the number of valid samples for property `i`.
- `r_i` is the numeric range used for property normalization.
- `y_i` is the ground-truth value.
- `ŷ_i` is the predicted value.

Evaluation outputs:

```text
wmae_details_sqrt.csv
wmae_contribution.png
wmae_contribution.pdf
```

---

## Recommended Workflow

### Step 1: Fine-tune the model

```bash
python fine_tuning.py
```

### Step 2: Run model inference

```bash
jupyter notebook infer.ipynb
```

### Step 3: Run Claude baseline inference

```bash
python claude_infer.py
```

### Step 4: Evaluate predictions

```bash
jupyter notebook wmae_eval_sqrt.ipynb
```

---

## Notes on Large Files

Model checkpoints, training outputs, prediction files, and logs can become large. These files should generally not be committed to GitHub.

Recommended `.gitignore`:

```text
.venv/
__pycache__/
.ipynb_checkpoints/

outputs/
runs/
checkpoints/
*.pt
*.pth
*.bin
*.safetensors

training_log.csv
3000outputs.txt
3000outputs_claude_haiku.txt
wmae_details_sqrt.csv
wmae_contribution.png
wmae_contribution.pdf

.env
```

---

## Security Notes

Before pushing this repository to GitHub:

- Remove all hard-coded API tokens.
- Remove private Hugging Face tokens from notebooks.
- Replace local absolute paths with relative paths.
- Do not commit model checkpoints unless intentionally releasing them.
- Use environment variables for all private credentials.

---

## Example Input and Output

### Input

```text
SMILES: C=C(C)OC
```

### Expected model behavior

```text
The monomer compound has sigma of ... GM at 780 nm, maximum sigma of ... GM, ISC of ... eV, toxicity score of ..., SA score of ..., boiling point of ... °C, logP of ..., aromaticity of ..., solubility of ... ug/mol, molecular weight of ... g/mol.
```

---

# Fine-tuning and evaluation:
Using QuantumChem-200K, we fine-tuned the open-source Qwen-2.5-32B LLM to create a chemistry AI assistant capable of forward polymer property prediction from SMILES. It demonstrates that domain-specific fine-tuning significantly improves prediction accuracy over baselines such as GPT-4o, Llama-3.1-70B, and the base Qwen2.5- 32B model. The evaluation metric used is the wMAE:

<img width="400" height="190" alt="Screenshot 2025-11-30 at 18 31 14" src="https://github.com/user-attachments/assets/d774aa09-5ea5-45fc-8f6d-11975669b065" />

# Benchmarking results:
<img width="11626" height="4706" alt="wmae_contribution_8models_rank_log" src="https://github.com/user-attachments/assets/01b6f116-2159-4438-a28e-d8ca47bb3451" />

<img width="5952" height="8418" alt="Radar_new" src="https://github.com/user-attachments/assets/ad94f91d-cb19-4c31-9146-311540a5bb54" />

## Citation

If you use this repository in academic work, please cite the associated project or manuscript.

```bibtex
@misc{quantumchem-zeng2026,
  title  = {QuantumChem-200K: A Large-Scale Organic Molecular Dataset for Quantum-Chemistry Property Screening and Language Model Benchmarking},
  author = {Yinqi Zeng, Shangding Gu, Jiancong Xiao, Ruizhong Qiu, Yuanchen Bei, Hanghang Tong, Renjie Li},
  year   = {2026},
  note   = {under review},
  url    = {https://arxiv.org/abs/2511.21747}
}
```
---
## License
Data: GNU General Public License v3, code: MIT.
