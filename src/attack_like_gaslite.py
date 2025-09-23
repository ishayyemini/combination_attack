import sys
import torch
from sentence_transformers import SentenceTransformer, util
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


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{dir_path}/GASLITE")

from GASLITE.src.attacks.gaslite import gaslite_attack
from GASLITE.src.full_attack import initialize_p_adv
from GASLITE.src.models.retriever import RetrieverModel

from attack import BlackBoxAttack

disable_progress_bars()
disable_progress_bar()

datasets = ["msmarco", "nq"]
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


def run_gaslite(model_hf_name, model, info, q, device):
    print("Running gaslite attack...")

    P_adv, trigger_slice, P_adv_before = initialize_p_adv(
        model=model,
        trigger_loc="suffix",
        trigger_len=72,
        adv_passage_init="dummy_token",
        mal_info=info,
        model_hf_name=model_hf_name,
    )
    P_adv = P_adv.to(device)
    emb_targets = model.embed(q).to(device)

    n_iter = 100
    n_grad = 5
    n_cand = 128
    n_flip = 20
    time_limit_in_seconds = None

    best_input_ids, out_metrics = gaslite_attack(
        model=model,
        # Passage to craft:
        trigger_slice=trigger_slice,
        inputs=P_adv,
        emb_targets=emb_targets,
        # Attack params:
        n_iter=n_iter,
        n_grad=n_grad,
        beam_search_config=dict(perform=True, n_cand=n_cand, n_flip=n_flip),
        time_limit_in_seconds=time_limit_in_seconds,
    )

    P_adv["input_ids"] = best_input_ids

    return model.tokenizer.decode(
        P_adv["input_ids"][0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Test attack like gaslight knows-all")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out", default="attack_like_gaslight")
    parser.add_argument("--gaslite", action="store_true", help="Run GASLITE attack")
    parser.add_argument(
        "--adv_dec", action="store_true", help="Run Adversarial Decoding attack"
    )
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


def load_model(model_hf_name, device, gaslite):
    if model_hf_name not in models:
        raise ValueError(f"Model {model_hf_name} not supported.")

    print(f"Loading model: {model_hf_name} on device: {device}")
    model = SentenceTransformer(model_hf_name).to(device)
    if gaslite:
        gaslite_model = RetrieverModel(
            model_hf_name=model_hf_name,
            sim_func_name=similarities[model_hf_name],
            device=device,
        )
    else:
        gaslite_model = None
    print("Model loaded successfully.")

    return model, gaslite_model


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
    filename = f"{dataset_name}-test_1.0_{model_hf_name.split('/')[1]}_{similarities[model_hf_name]}.json"
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
        return util.cos_sim
    else:
        return util.dot_score


def run_attack(model_hf_name, dataset_name, trials, device, index, gaslite, adv_dec):
    print(
        f"Running attack on model: {model_hf_name}, dataset: {dataset_name}, index: {index}"
    )

    model, gaslite_model = load_model(model_hf_name, device, gaslite)
    corpus, queries = load_ds(dataset_name, model_hf_name)
    results = load_results(dataset_name, model_hf_name)

    sim_kind = similarities[model_hf_name]
    sim = similarity_fn(sim_kind)

    chosen_qids = np.random.choice(list(queries.keys()), size=(trials,), replace=False)

    info_rankings = []
    stuffing_rankings = []
    adv_rankings = []
    gaslite_rankings = []
    adv_dec_rankings = []

    best_similarities = []
    info_similarities = []
    stuffing_similarities = []
    adv_similarities = []
    gaslite_similarities = []
    adv_dec_similarities = []

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
        print("Performing black-box attack...")
        bb_attack = BlackBoxAttack(model, q, sim=sim)
        tokens, _, _ = bb_attack.combination_attack(
            info,
            p_init=0.3946,
            total_tokens=72,
            num_iters=2940,
            random_pool_per_pos=316,
        )

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

        if gaslite:
            p_gaslite = run_gaslite(model_hf_name, gaslite_model, info, q, device)
            print(f"gaslite: {p_gaslite}")

            # calculate gaslite similarity
            with torch.inference_mode():
                p_gaslite_enc = model.encode(p_gaslite, convert_to_tensor=True)
            gaslite_sim = sim(q_enc, p_gaslite_enc).item()
            print(f"Similarity between query and GASLITE passage: {gaslite_sim}")

            # store gaslite similarity and ranking
            gaslite_similarities.append(gaslite_sim)
            gaslite_rankings.append(calc_ranking(gaslite_sim, qid, results))

        if adv_dec:
            print("Performing adv_dec attack...")
            doc_ids = list(results[qid].keys())[:10]
            reference_texts = [corpus[d]["text"] for d in doc_ids]
            s_tokens, _, _ = bb_attack.adversarial_decoding_rag(
                info,
                reference_texts=reference_texts,
                max_tokens=57,
                beam_size=9,
                pool_size=208,
                sample_per_beam=104,
                temperature=0.9752,
                verbose=False,
            )
            p_adv_dec = info + " " + " ".join(s_tokens)
            print(f"adv_dec: {p_adv_dec}")

            # calculate adversarial decoding similarity
            with torch.inference_mode():
                p_adv_dec_enc = model.encode(p_adv_dec, convert_to_tensor=True)
            adv_dec_sim = sim(q_enc, p_adv_dec_enc).item()
            print(f"Similarity between query and adv_dec passage: {adv_dec_sim}")

            # store adversarial decoding similarity and ranking
            adv_dec_similarities.append(adv_dec_sim)
            adv_dec_rankings.append(calc_ranking(adv_dec_sim, qid, results))

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

    if gaslite:
        record["gaslite_appeared_1"] = sum(r == 0 for r in gaslite_rankings) / trials
        record["gaslite_appeared_5"] = sum(r < 5 for r in gaslite_rankings) / trials
        record["gaslite_appeared_10"] = sum(r < 10 for r in gaslite_rankings) / trials
        record["gaslite_sim"] = sum(gaslite_similarities) / trials

    if adv_dec:
        record["adv_dec_appeared_1"] = sum(r == 0 for r in adv_dec_rankings) / trials
        record["adv_dec_appeared_5"] = sum(r < 5 for r in adv_dec_rankings) / trials
        record["adv_dec_appeared_10"] = sum(r < 10 for r in adv_dec_rankings) / trials
        record["adv_dec_sim"] = sum(adv_dec_similarities) / trials

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
    for t in my_tasks:
        print(f"SLURM rank {rank} will do task: {t}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    records = []
    for model_hf_name, dataset_name, i in my_tasks:
        rec = run_attack(
            model_hf_name=model_hf_name,
            dataset_name=dataset_name,
            trials=args.trials,
            device=device,
            index=i,
            gaslite=args.gaslite,
            adv_dec=args.adv_dec,
        )
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
