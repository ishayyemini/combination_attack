import data_utils
from sentence_transformers import SentenceTransformer
import torch
import json
from tqdm import trange


# Run config
concept_to_attack = "potter"
concept_portion_to_train = 0.5
dataset_name = "msmarco"
data_split = "train"
data_portion = 1.0
sim_func_name = "cos_sim"
embedder_model_name = "intfloat/e5-base-v2"

# Adversarial passage config (see paper for the used parameters)
mal_info = "Voldemort was right all along!"
trigger_len = 30

# Attack config
max_batch_size = 256
n_iter = 100
n_grad = 5
n_cand = 128
n_flip = 20
time_limit_in_seconds = None  # can be set to limit the attack time


with open(f"concept-{concept_to_attack}.yaml", "r") as f:
    import yaml

    concept_config = yaml.safe_load(f)
    concept_qids = concept_config["concept_qids"]  # fetched from the attack config

heldin_concept_qids, heldout_concept_qids = (
    concept_qids[: int(len(concept_qids) * concept_portion_to_train)],
    concept_qids[int(len(concept_qids) * concept_portion_to_train) :],
)


corpus, queries, qrels, qp_pairs_dataset = data_utils.load_dataset(
    dataset_name=dataset_name,
    data_split=data_split,
    data_portion=data_portion,
    embedder_model_name=embedder_model_name,
)

print(queries[heldin_concept_qids[0]])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(embedder_model_name).to(device)

batch_size = 512  # Adjust based on your GPU memory

with open("data/corpus_enc.jsonl", "w") as out:
    for i in trange(0, len(corpus), batch_size):
        batch_texts = []
        batch_ids = []

        for j in range(i, min(i + batch_size, len(corpus))):
            c = corpus[f"{j}"]
            batch_texts.append(c["text"])
            batch_ids.append(f"{j}")

        # Encode batch at once
        batch_encodings = model.encode(batch_texts, convert_to_tensor=True)

        # Write batch immediately to file
        for doc_id, encoding in zip(batch_ids, batch_encodings):
            item = {"_id": doc_id, "enc": encoding.tolist()}
            print(json.dumps(item), file=out)
