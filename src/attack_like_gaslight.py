import torch
from sentence_transformers import SentenceTransformer
import data_utils
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
import json
import numpy as np
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from tqdm import tqdm
import argparse
import pandas as pd
from pathlib import Path
import os

from attack import BlackBoxAttack

disable_progress_bars()
disable_progress_bar()

datasets = {"msmarco", "nq"}
models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "Snowflake/snowflake-arctic-embed-m",
    "intfloat/e5-base-v2",
    "sentence-transformers/gtr-t5-base",
    "facebook/contriever",
    "facebook/contriever-msmarco",
    "sentence-transformers/msmarco-roberta-base-ance-firstp",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
]
datasets_models = {"msmarco": [0, 1, 2, 3, 4, 5, 6, 7, 8], "nq": [0, 3, 6, 7]}
splits = {"msmarco": "dev", "nq": "test"}
similarities = {
    "sentence-transformers/all-MiniLM-L6-v2": "cos_sim",
    "sentence-transformers/all-mpnet-base-v2": "cos_sim",
    "Snowflake/snowflake-arctic-embed-m": "cos_sim",
    "intfloat/e5-base-v2": "cos_sim",
    "sentence-transformers/gtr-t5-base": "cos_sim",
    "facebook/contriever": "dot",
    "facebook/contriever-msmarco": "dot",
    "sentence-transformers/msmarco-roberta-base-ance-firstp": "dot",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": "dot",
}
toxic_prefixes = []


def parse_args():
    parser = argparse.ArgumentParser(description="Test attack like gaslight knows-all")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out", default="attack_like_gaslight")
    return parser.parse_args()


def get_toxic_passage(model_hf_name):
    if len(toxic_prefixes) == 0:
        toxic_prefixes.extend(
            [
                row["text"]
                for row in load_dataset("toxigen/toxigen-data")["train"]
                if row["toxicity_ai"] == 5 and 50 < len(row["text"]) < 100
            ]
        )
    passage = np.random.choice(toxic_prefixes).strip()
    if model_hf_name.endswith("e5-base-v2"):
        passage = "passage: " + passage
    return passage


def load_model(model_hf_name, device):
    if model_hf_name not in models:
        raise ValueError(f"Model {model_hf_name} not supported.")

    print(f"Loading model: {model_hf_name} on device: {device}")
    model = SentenceTransformer(model_hf_name).to(device)
    print("Model loaded successfully.")
    return model


def load_ds(dataset_name, model_hf_name):
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    if models.index(model_hf_name) not in datasets_models[dataset_name]:
        raise ValueError(f"Model {model_hf_name} not supported for {dataset_name}.")

    print(f"Loading dataset: {dataset_name} ({splits[dataset_name]})")
    corpus, queries, _, _ = data_utils.load_dataset(
        dataset_name=dataset_name,
        data_split=splits[dataset_name],
        embedder_model_name=model_hf_name,
    )
    print(f"Loaded {len(corpus)} corpus items, {len(queries)} queries.")
    return corpus, queries


def load_results(dataset_name, model_hf_name):
    filename = f"{dataset_name}-test_1.0_{model_hf_name.split("/")[1]}_{similarities[model_hf_name]}.json"
    print(f"Downloading results from HuggingFace Hub: {filename}")
    try:
        local_results_path = hf_hub_download(
            repo_id="MatanBT/retrieval-datasets-similarities",
            filename=filename,
            repo_type="dataset",
        )
        print(f"Results downloaded to: {local_results_path}")
    except Exception as e:
        raise ValueError(f"Could not download results file: {e}")

    print("Loading results from:", local_results_path)
    with open(local_results_path) as f:
        results = json.load(f)
    print("Results loaded successfully.")
    return results


def calc_ranking(sim, qid, results):
    ranking = 0
    for pid, score in results[qid].items():
        if score > sim:
            ranking += 1
    return ranking


def similarity_fn(kind):
    if kind == "cos_sim":
        return torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    else:
        return lambda x, y: torch.dot(x, y)


def run_attack(model_hf_name, dataset_name, trials, device, index):
    print(
        f"Running attack on model: {model_hf_name}, dataset: {dataset_name}, index: {index}"
    )

    model = load_model(model_hf_name, device)
    corpus, queries = load_ds(dataset_name, model_hf_name)
    results = load_results(dataset_name, model_hf_name)

    sim_kind = similarities[model_hf_name]
    sim = similarity_fn(sim_kind)

    chosen_qids = np.random.choice(list(queries.keys()), size=(trials,), replace=False)

    info_rankings = []
    stuffing_rankings = []
    adv_rankings = []

    best_similarities = []
    info_similarities = []
    stuffing_similarities = []
    adv_similarities = []

    for i, qid in enumerate(tqdm(chosen_qids, desc=f"Trials {index}")):
        print(f"Trial number {i + 1} inside attack {index}")

        best_pid = list(results[qid].keys())[0]

        # original passage and query
        q = queries[qid]
        print(f"{qid}: {q}")
        p = corpus[best_pid]["text"]
        print(f"{best_pid}: {p}")
        info = get_toxic_passage(model_hf_name)
        print(f"info: {info}")

        # batch encodings (except adv which depends on attack)
        base_texts = [
            q,
            p,
            info,
            info + " " + q.replace("query: ", ""),
        ]  # q, best passage, info, stuffing
        with torch.inference_mode():
            encs = model.encode(base_texts, convert_to_tensor=True)
        q_enc, p_enc, info_enc, stuffing_enc = encs

        # calculate similarities
        best_sim = sim(q_enc, p_enc).item()
        print(f"Similarity between query and original best passage: {best_sim}")
        info_sim = sim(q_enc, info_enc).item()
        print(f"Similarity between query and info passage: {info_sim}")
        stuffing_sim = sim(q_enc, stuffing_enc).item()
        print(f"Similarity between query and stuffing passage: {stuffing_sim}")

        # store similarities
        best_similarities.append(best_sim)
        info_similarities.append(info_sim)
        stuffing_similarities.append(stuffing_sim)

        # store rankings
        info_rankings.append(calc_ranking(info_sim, qid, results))
        stuffing_rankings.append(calc_ranking(stuffing_sim, qid, results))

        # perform attack
        bb_attack = BlackBoxAttack(model, q)
        tokens, _, _ = bb_attack.combination_attack(
            info,
            p_init=0.3946,
            total_tokens=72,
            num_iters=2940,
            random_pool_per_pos=316,
        )

        # doc_ids = list(results[qid].keys())[:10]
        # reference_texts = [corpus[d]["text"] for d in doc_ids]
        # s_tokens, _, _ = bb_attack.adversarial_decoding_rag(
        #     info,
        #     reference_texts=reference_texts,
        #     max_tokens=57,
        #     beam_size=9,
        #     pool_size=208,
        #     sample_per_beam=104,
        #     temperature=0.9752,
        #     verbose=True,
        # )

        # craft adversarial passage
        p_adv = info + " " + " ".join(tokens)
        print(f"adversarial: {p_adv}")

        # calculate adv similarity
        with torch.inference_mode():
            p_adv_enc = model.encode(p_adv, convert_to_tensor=True)
        adv_sim = sim(q_enc, p_adv_enc).item()
        print(f"Similarity between query and attacked passage: {adv_sim}")

        # store adv similarity and ranking
        adv_similarities.append(adv_sim)
        adv_rankings.append(calc_ranking(adv_sim, qid, results))
        print("\n", flush=True)

    record = {
        "model": model_hf_name,
        "dataset": dataset_name,
        "similarity": sim_kind,
        "trials": trials,
        "info_appeared_1": sum(r == 0 for r in info_rankings) / trials,
        "info_appeared_5": sum(r < 5 for r in info_rankings) / trials,
        "info_appeared_10": sum(r < 10 for r in info_rankings) / trials,
        "info_sim": sum(info_similarities) / trials,
        "stuffing_appeared_1": sum(r == 0 for r in stuffing_rankings) / trials,
        "stuffing_appeared_5": sum(r < 5 for r in stuffing_rankings) / trials,
        "stuffing_appeared_10": sum(r < 10 for r in stuffing_rankings) / trials,
        "stuffing_sim": sum(stuffing_similarities) / trials,
        "adv_appeared_1": sum(r == 0 for r in adv_rankings) / trials,
        "adv_appeared_5": sum(r < 5 for r in adv_rankings) / trials,
        "adv_appeared_10": sum(r < 10 for r in adv_rankings) / trials,
        "adv_sim": sum(adv_similarities) / trials,
    }

    print()
    print(f"Rankings of original passages: {info_rankings}")
    print(f"Rankings of stuffing passages: {stuffing_rankings}")
    print(f"Rankings of attacked passages: {adv_rankings}")
    print(
        f"Average ranking of attacked passages: {sum(adv_rankings) / len(adv_rankings)}"
    )
    print(
        f"Top 1 in {sum(r == 0 for r in adv_rankings)} out of {len(adv_rankings)} attacks!"
    )
    print(
        f"Top 5 in {sum(r < 5 for r in adv_rankings)} out of {len(adv_rankings)} attacks!"
    )
    print(
        f"Top 10 in {sum(r < 10 for r in adv_rankings)} out of {len(adv_rankings)} attacks!"
    )

    print()
    print(f"Similarities of best passages: {best_similarities}")
    print(f"Similarities of original passages: {info_similarities}")
    print(f"Similarities of stuffing passages: {stuffing_similarities}")
    print(f"Similarities of attacked passages: {adv_similarities}")
    print(
        f"Average similarity of attacked passages: {sum(adv_similarities) / len(adv_similarities)}"
    )
    print("\n\n\n\n", flush=True)

    return record


def build_tasks():
    tasks = []
    i = 0
    for dataset_name in datasets:
        for model_idx in datasets_models[dataset_name]:
            tasks.append((models[model_idx], dataset_name, i))
            i += 1
    return tasks


def main():
    args = parse_args()
    out_path = Path(args.out).with_suffix(".csv")

    all_tasks = build_tasks()

    # SLURM environment
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))

    my_tasks = all_tasks[rank::world_size]
    print(
        f"SLURM rank {rank}/{world_size} handling {len(my_tasks)} of {len(all_tasks)} tasks"
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    records = []
    for model_hf_name, dataset_name, i in my_tasks:
        rec = run_attack(model_hf_name, dataset_name, args.trials, device, i)
        records.append(rec)
        # Write (append or create)
        if out_path.exists():
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
            combined.drop_duplicates(
                subset=["model", "dataset", "trials"], keep="last"
            ).to_csv(out_path, index=False)
        else:
            pd.DataFrame(records).to_csv(out_path, index=False)
        records = []  # flush after write to avoid duplication

    print(f"done on rank {rank}")


if __name__ == "__main__":
    main()
