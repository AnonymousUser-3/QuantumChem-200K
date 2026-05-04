import os
import pandas as pd
from anthropic import Anthropic

# =========================
# Claude client
# =========================
client = Anthropic(
    api_key=os.getenv("your api key")  # or hard-code
)

# =========================
# Prompt template (unchanged)
# =========================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction = "Based on the given SMILES string, predict the monomer's relevant properties. Given toxicity and SA score ranging from 0 to 1, aromaticity can only take 0 or 1. Answer the task in this format: The monomer compound has sigma of ? GM at 780 nm, maximum sigma of ? GM, ISC of ? eV, toxicity score of ?, SA score of ?, boiling point of ? °C, logP of ?, aromaticity of ?, solubility of ? ug/mol, molecular weight of ? g/mol."


# =========================
# Single inference test
# =========================
smiles = "CC12NC1C1C(C#N)CF"
prompt = alpaca_prompt.format(instruction, smiles, "")

resp = client.messages.create(
    model="claude-haiku-4.5",
    max_tokens=256,
    temperature=0.2,
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print(resp.content[0].text)

# =========================
# Batch inference
# =========================
file_path = "3000.csv"
smiles_df = pd.read_csv(file_path)

all_outputs = []
print("3000.csv loaded")

for i, smi in enumerate(smiles_df["SMILES"].head(3000)):
    prompt = alpaca_prompt.format(instruction, smi, "")
    print(smi)

    resp = client.messages.create(
        model="claude-haiku-4.5",
        max_tokens=256,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    decoded = resp.content[0].text
    all_outputs.append((smi, decoded))

    print(f"Completed {i+1}/3000")

# =========================
# Save results
# =========================
with open("3000outputs_claude_haiku.txt", "w", encoding="utf-8") as f:
    for smi, out in all_outputs:
        f.write(f"input smiles: {smi}\n")
        f.write(out.strip() + "\n\n")
