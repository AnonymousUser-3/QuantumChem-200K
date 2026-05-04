# conda install gcc

from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel, FastModel
import torch

BASE_MODEL_ID   = "unsloth/Qwen2.5-32B"          # your base
LORA_ADAPTER_ID = "YinqiZeng704/gemma_ablation"         # <<— change this

HF_TOKEN = "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"


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    model_name = LORA_ADAPTER_ID, #or the Qwen2.5 model, or the GPT-4 model (API key required), or any other models you prefer
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = HF_TOKEN, # use one if using gated models like meta-llama/Llama-2-7b-hf
)

#need for image-text input models
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

prompt = alpaca_prompt.format(
    "Based on the given SMILES string, predict the monomer's relevant properties. Given toxicity and SA score ranging from 0 to 1, aromaticity can only take 0 or 1. Answer the task in this format: The monomer compound has sigma of ? GM at 780 nm, maximum sigma of ? GM, ISC of ? eV, toxicity score of ?, SA score of ?, boiling point of ? °C, logP of ?, aromaticity of ?, solubility of ? ug/mol, molecular weight of ? g/mol.", # instruction
    "CC12NC1C1C(C#N)C21",
    ""
)

# IMPORTANT: use text= ... not positional
inputs = tokenizer(text=[prompt], return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

import pandas as pd
# === Load SMILES ===
file_path = "3000.csv"
smiles_df = pd.read_csv(file_path)
print("3000.csv loaded")

all_outputs = []  # Initialize list to store the outputs

for i, smi in enumerate(smiles_df["SMILES"].head(3000)):
    prompt = alpaca_prompt.format(
        #"Based on the given SMILES string, predict the monomer's relevant properties. Given toxicity and SA score ranging from 0 to 1, aromaticity can only take 0 or 1. Answer the task in this format: The monomer compound has sigma of ? GM at 780 nm, maximum sigma of ? GM, ISC of ? eV, toxicity score of ?, SA score of ?, boiling point of ? °C, logP of ?, aromaticity of ?, solubility of ? ug/mol, molecular weight of ? g/mol.",  # instruction
        "Based on the given SMILES string, predict the monomer's relevant properties.",
        smi,
        ""  # leave blank for generation
    )
    print(smi)

    # IMPORTANT: Gemma3Processor expects text=... (otherwise your prompt is treated as images)
    inputs = tokenizer(text=[prompt], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)

    # batch_decode exists on many tokenizers; Gemma3Processor may not always expose it
    # so decode via the underlying tokenizer if needed.
    try:
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    except AttributeError:
        decoded = tokenizer.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    all_outputs.append(decoded)

    print(f"Completed {i+1}/3000")  # progress bar in notebook

# === Save results ===
with open("3000outputs_gemma_finetuned.txt", "w", encoding="utf-8") as f:
    for out in all_outputs:
        f.write(out + "\n\n")
