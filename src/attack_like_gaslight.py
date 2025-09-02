import torch
from sentence_transformers import SentenceTransformer
import data_utils
from huggingface_hub import hf_hub_download
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from attack import BlackBoxAttack


toxic_dataset = load_dataset("toxigen/toxigen-data")["train"]
toxic_prefixes = [
    row["text"]
    for row in toxic_dataset
    if row["toxicity_ai"] == 5 and 50 < len(row["text"]) < 100
]  # filter to highly toxic text, with reasonable length


dataset_name = "msmarco"
data_split = "dev"
data_portion = 1.0
embedder_model_name = "intfloat/e5-base-v2"

cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

print(
    f"Loading dataset: {dataset_name} ({data_split}) with {data_portion * 100}% of data"
)
corpus, queries, qrels, qp_pairs_dataset = data_utils.load_dataset(
    dataset_name=dataset_name,
    data_split=data_split,
    data_portion=data_portion,
    embedder_model_name=embedder_model_name,
)
print(
    f"Loaded {len(corpus)} corpus items, {len(queries)} queries, and {len(qrels)} qrels."
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model: {embedder_model_name} on device: {device}")
model = SentenceTransformer(embedder_model_name).to(device)
print("Model loaded successfully.")

filename = "msmarco-test_1.0_e5-base-v2_cos_sim.json"
print(f"Downloading results from HuggingFace Hub: {filename}")
local_results_path = hf_hub_download(
    repo_id="MatanBT/retrieval-datasets-similarities",
    filename=filename,
    repo_type="dataset",
)
print(f"Results downloaded to: {local_results_path}")

print("Loading results from:", local_results_path)
with open(local_results_path) as f:
    results = json.load(f)
print("Results loaded successfully.")


test_size = 50

chosen_queries = np.random.choice(
    list(queries.keys()), size=(test_size,), replace=False
)

orig_rankings = []
stuffing_rankings = []
attacked_rankings = []

best_similarities = []
orig_similarities = []
stuffing_similarities = []
attacked_similarities = []

for i, qid in enumerate(tqdm(chosen_queries)):
    q = queries[qid]
    p = corpus[list(results[qid].keys())[0]]["text"]
    p_adv = "passage: " + np.random.choice(toxic_prefixes).strip()

    print(f"{qid}: {q}")
    print(f"{list(results[qid].keys())[0]}: {p}")
    print(p_adv)

    q_enc = model.encode(q, convert_to_tensor=True)
    p_enc = model.encode(p, convert_to_tensor=True)
    p_adv_enc = model.encode(p_adv, convert_to_tensor=True)
    stuffing_enc = model.encode(
        p_adv + " " + q.replace("query: ", ""), convert_to_tensor=True
    )

    best_sim = cos_sim(q_enc, p_enc)
    print(f"Similarity between query and original best passage: {best_sim}")
    adv_sim = cos_sim(q_enc, p_adv_enc)
    print(f"Similarity between query and adversarial passage: {adv_sim}")
    stuffing_sim = cos_sim(q_enc, stuffing_enc)
    print(f"Similarity between query and stuffing passage: {stuffing_sim}")

    best_similarities.append(best_sim.item())
    orig_similarities.append(adv_sim.item())
    stuffing_similarities.append(stuffing_sim.item())

    orig_ranking = 0
    for pid, score in results[qid].items():
        if score > adv_sim:
            orig_ranking += 1
    orig_rankings.append(orig_ranking)

    stuffing_ranking = 0
    for pid, score in results[qid].items():
        if score > stuffing_sim:
            stuffing_ranking += 1
    stuffing_rankings.append(stuffing_ranking)

    bb_attack = BlackBoxAttack(model, q)

    print("Attacking after stuffing the query...")
    p_adv += " " + q.replace("query: ", "")

    print()
    print(f"Attack number {i + 1}")
    # tokens = bb_attack.attack(p_adv)
    # s_tokens, current_prompt, history = bb_attack.square_attack(
    #     p_adv,
    #     total_tokens=60,
    #     num_iters=1500,
    #     random_pool_per_pos=200,
    #     early_stop_patience=100,
    #     initial_tokens=tokens,
    # )

    doc_ids = list(results[qid].keys())[:10]
    reference_texts = [corpus[d]["text"] for d in doc_ids]
    s_tokens, _, _ = bb_attack.adversarial_decoding_rag(
        p_adv,
        reference_texts=reference_texts,
        max_tokens=57,
        beam_size=9,
        pool_size=208,
        sample_per_beam=104,
        temperature=0.9752,
        verbose=True,
    )

    p_attacked = p_adv + " " + " ".join(s_tokens)
    print(f"Adversarial passage: {p_attacked}")

    p_attacked_enc = model.encode(p_attacked, convert_to_tensor=True)
    attacked_sim = cos_sim(q_enc, p_attacked_enc)
    print(f"Similarity between query and attacked passage: {attacked_sim}")
    attacked_similarities.append(attacked_sim.item())

    attacked_ranking = 0
    for pid, score in results[qid].items():
        if score > attacked_sim:
            attacked_ranking += 1
    attacked_rankings.append(attacked_ranking)
    print("\n\n", flush=True)

print(f"Rankings of original passages: {orig_rankings}")
print(f"Rankings of stuffing passages: {stuffing_rankings}")
print(f"Rankings of attacked passages: {attacked_rankings}")
print(
    f"Average ranking of attacked passages: {sum(attacked_rankings) / len(attacked_rankings)}"
)
print(
    f"Top 1 in {sum(r == 0 for r in attacked_rankings)} out of {len(attacked_rankings)} attacks!"
)
print(
    f"Top 10 in {sum(r < 10 for r in attacked_rankings)} out of {len(attacked_rankings)} attacks!"
)

print()
print(f"Similarities of best passages: {best_similarities}")
print(f"Similarities of original passages: {orig_similarities}")
print(f"Similarities of stuffing passages: {stuffing_similarities}")
print(f"Similarities of attacked passages: {attacked_similarities}")
print(
    f"Average similarity of attacked passages: {sum(attacked_similarities) / len(attacked_similarities)}"
)
