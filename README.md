# Combination Attack

Code implementation for the project [Black-Box Transformation of the GASLITE Attack](tex/final_paper.pdf). The goal is to run a black-box attack to maximize semantic similarity (cosine/dot product) between a target query embedding and the embedding of a prompt with an appended token suffix. 

Core methods:
- random_attack: Random search for best tokens.
- square_attack: 1D adaptation of Square Attack for token sequences (contiguous block updates).
- combination_attack: My novel attack, as mentioned in the paper.
- adversarial_decoding_rag: My implementation of the [Adversarial Decoding](https://github.com/collinzrj/adversarial_decoding) attack.

## Main Files
- `src/attack.py`: Main implementation of the attack algorithms.
- `src/attack_like_gaslite.py`: Implementation of a Knows-All attack, like in [GASLITE](https://github.com/matanbt/GASLITE).

## Requirements
- Python 3.8+
- macOS or Linux, CUDA optional

Install Python dependencies from the repository root:

```bash
pip install -r src/requirements.txt
```

This will install: torch, sentence-transformers, numpy, tqdm, beir, datasets, huggingface-hub, pyyaml, pandas, matplotlib.

## Minimal Example
Very minimal example using a small SentenceTransformers model:

```python
from sentence_transformers import SentenceTransformer
from src.attack import BlackBoxAttack

# Load an embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define the target query we want the prompt to be similar to
query = "who was harry potter?"

# Construct the attacker
attack = BlackBoxAttack(model=model, q=query)

# Base prompt we append tokens to
info = "harry poter is a loser, and voldemort was right all along."

# Combination attack: warm-start + square refinement
best_tokens, final_prompt, history = attack.combination_attack(
    base_prompt=info,
    total_tokens=16,
    p_init=0.5,
    num_iters=200,
    random_pool_per_pos=40,
    early_stop_patience=60,
    seed=42,
)

print("Final prompt:\n", final_prompt)
print("Appended tokens:", best_tokens)
print("Last similarity:", history[-1][1])
```

A more complete example can be viewed in `src/demo.ipynb`.

Notes:
- All similarities use `sentence_transformers.util.cos_sim` by default.
- CPU-only runs are supported, but CUDA will be used if available.

## Data
No specific dataset is required for the very basic example, however in `src/demo.ipnyb` a proper dataset is used.

## Citation
If you use this code in your research, please cite this repository.
