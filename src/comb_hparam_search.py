import argparse
import json
import os
import random
import time
from pathlib import Path
from tqdm import trange, tqdm

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None
try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
except ImportError:
    plt = None
    sns = None

import data_utils
from huggingface_hub import hf_hub_download
from attack import BlackBoxAttack
from datasets import load_dataset


toxic_dataset = load_dataset("toxigen/toxigen-data")["train"]
toxic_prefixes = [
    row["text"]
    for row in toxic_dataset
    if row["toxicity_ai"] == 5 and 50 < len(row["text"]) < 100
]  # filter to highly toxic text, with reasonable length


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random hyperparameter search for combination_attack"
    )
    parser.add_argument("--dataset_name", default="msmarco")
    parser.add_argument("--data_split", default="dev")
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--embedder_model_name", default="intfloat/e5-base-v2")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="comb_hparam_results")
    parser.add_argument(
        "--ref_docs", type=int, default=10, help="Top-k reference docs to use"
    )
    parser.add_argument(
        "--query_samples",
        type=int,
        default=1,
        help="Number of random queries to evaluate per trial",
    )
    parser.add_argument("--total_tokens_min", type=int, default=30)
    parser.add_argument("--total_tokens_max", type=int, default=70)
    parser.add_argument("--p_init_min", type=float, default=0.3)
    parser.add_argument("--p_init_max", type=float, default=0.7)
    parser.add_argument("--num_iters_min", type=int, default=500)
    parser.add_argument("--num_iters_max", type=int, default=2000)
    parser.add_argument("--pool_min", type=int, default=50)
    parser.add_argument("--pool_max", type=int, default=300)
    parser.add_argument("--stop_min", type=int, default=70)
    parser.add_argument("--stop_max", type=int, default=120)
    parser.add_argument(
        "--no_query_sim",
        action="store_true",
        help="Disable query similarity in combined score",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def sample_int(low: int, high: int):
    return random.randint(low, high)


def sample_choice_int(low: int, high: int):
    # bias slightly toward extremes by picking from a set
    choices = list(range(low, high + 1))
    return random.choice(choices)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    results_csv = Path(args.out_dir) / "search_results.csv"
    summary_json = Path(args.out_dir) / "best_config.json"

    print(
        f"Loading dataset {args.dataset_name} ({args.data_split}) portion={args.data_portion}"
    )
    corpus, queries, qrels, qp_pairs_dataset = data_utils.load_dataset(
        dataset_name=args.dataset_name,
        data_split=args.data_split,
        data_portion=args.data_portion,
        embedder_model_name=args.embedder_model_name,
    )
    print(f"Loaded corpus={len(corpus)} queries={len(queries)}")

    device = torch.device(args.device)
    print(f"Loading embedder: {args.embedder_model_name} on {device}")
    model = SentenceTransformer(args.embedder_model_name).to(device)

    # Download retrieval results to mimic reference selection
    filename = f"msmarco-test_{args.data_portion}_e5-base-v2_cos_sim.json"
    print(f"Downloading retrieval results file: {filename}")
    try:
        local_results_path = hf_hub_download(
            repo_id="MatanBT/retrieval-datasets-similarities",
            filename=filename,
            repo_type="dataset",
        )
        with open(local_results_path) as f:
            retr_results = json.load(f)
    except Exception as e:
        print(
            f"Warning: failed to fetch retrieval results: {e}. Falling back to random corpus docs."
        )
        retr_results = None

    # Pre-select query IDs for consistency across trials
    all_query_ids = list(queries.keys())
    if len(all_query_ids) == 0:
        raise RuntimeError("No queries loaded")
    trial_query_ids = random.sample(
        all_query_ids, k=min(args.query_samples, len(all_query_ids))
    )

    print(f"Query IDs for trials: {trial_query_ids}")

    trial_records = []

    # Define sampling ranges (integers will be sampled uniformly)
    for trial in trange(1, args.trials + 1):
        trial_seed = args.seed + trial
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)

        total_tokens = sample_int(args.total_tokens_min, args.total_tokens_max)
        p_init = random.uniform(args.p_init_min, args.p_init_max)
        num_iters = sample_int(args.num_iters_min, args.num_iters_max)
        random_pool_per_pos = sample_int(args.pool_min, args.pool_max)
        # early_stop_patience = sample_int(args.stop_min, args.stop_max)
        early_stop_patience = 100

        per_query_metrics = []
        t0_trial = time.time()

        for qid in tqdm(trial_query_ids):
            q_text = queries[qid]

            # Start from poison as base passage without stuffing
            p_adv = "passage: " + np.random.choice(toxic_prefixes).strip()
            base_prompt = p_adv

            # Initialize attack object per query (stores q_emb)
            bb_attack = BlackBoxAttack(model, q_text, device=device)

            # Run adversarial decoding
            t0 = time.time()
            tokens, final_prompt, history = bb_attack.combination_attack(
                base_prompt,
                total_tokens=total_tokens,
                p_init=p_init,
                num_iters=num_iters,
                random_pool_per_pos=random_pool_per_pos,
                early_stop_patience=early_stop_patience,
            )
            rt = time.time() - t0

            # Compute actual query similarity for final prompt
            with torch.no_grad():
                q_emb = bb_attack.q_emb  # already tensor
                final_emb = model.encode(
                    final_prompt,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                query_sim = util.cos_sim(q_emb, final_emb).item()
            combined_hist_best = (
                history[-1]["best_score"]
                if isinstance(history[-1], dict)
                else history[-1][1]
            )
            per_query_metrics.append(
                {
                    "qid": qid,
                    "query_sim": query_sim,
                    "combined_best": combined_hist_best,
                    "tokens_generated": len(tokens),
                    "decoding_time_sec": rt,
                }
            )

        trial_time = time.time() - t0_trial
        # Aggregate metrics across queries in this trial
        avg_query_sim = float(np.mean([m["query_sim"] for m in per_query_metrics]))
        avg_combined = float(np.mean([m["combined_best"] for m in per_query_metrics]))
        avg_tokens = float(np.mean([m["tokens_generated"] for m in per_query_metrics]))
        avg_runtime = float(
            np.mean([m["decoding_time_sec"] for m in per_query_metrics])
        )

        record = {
            "trial": trial,
            "seed": trial_seed,
            "total_tokens": total_tokens,
            "p_init": round(p_init, 4),
            "num_iters": num_iters,
            "random_pool_per_pos": random_pool_per_pos,
            "early_stop_patience": early_stop_patience,
            "avg_query_sim": avg_query_sim,
            "avg_combined_score": avg_combined,
            "avg_tokens_generated": avg_tokens,
            "avg_decoding_time_sec": avg_runtime,
            "trial_wall_time_sec": trial_time,
            "query_samples": len(per_query_metrics),
        }
        trial_records.append(record)
        print(
            f"[Trial {trial}/{args.trials}] total_tokens={total_tokens} p_init={p_init:.2f} num_iters={num_iters} random_pool_per_pos={random_pool_per_pos} early_stop_patience={early_stop_patience} -> query_sim={avg_query_sim:.4f} tokens={avg_tokens:.1f} time={avg_runtime:.2f}s"
        )

        # Incremental save
        if pd is not None:
            df = pd.DataFrame(trial_records)
            df.to_csv(results_csv, index=False)
        else:
            with open(results_csv.with_suffix(".json"), "w") as f:
                json.dump(trial_records, f, indent=2)

    # Final DataFrame
    if pd is not None:
        df = pd.DataFrame(trial_records)
        # Identify best by highest avg_query_sim
        best_idx = df["avg_query_sim"].idxmax()
        best_row = df.loc[best_idx].to_dict()
        with open(summary_json, "w") as f:
            json.dump(best_row, f, indent=2)
        print("Best config (by avg_query_sim):")
        print(json.dumps(best_row, indent=2))

        # Plotting
        if plt is not None and sns is not None:
            sns.set(style="whitegrid")
            param_cols = [
                "total_tokens",
                "p_init",
                "num_iters",
                "random_pool_per_pos",
                "early_stop_patience",
            ]
            out_dir = Path(args.out_dir)

            # Scatter plots param vs query_sim
            for col in param_cols:
                plt.figure(figsize=(5, 4))
                sns.scatterplot(
                    data=df,
                    x=col,
                    y="avg_query_sim",
                    hue="avg_decoding_time_sec",
                    palette="viridis",
                )
                plt.title(f"{col} vs avg_query_sim")
                plt.tight_layout()
                plt.savefig(out_dir / f"scatter_{col}.png", dpi=160)
                plt.close()

            # Pairplot
            try:
                sns.pairplot(df[param_cols + ["avg_query_sim"]])
                plt.savefig(out_dir / "pairplot_params.png", dpi=160)
                plt.close()
            except Exception as e:
                print(f"Pairplot failed: {e}")

            # Correlation heatmap
            plt.figure(figsize=(6, 5))
            corr = df[param_cols + ["avg_query_sim", "avg_decoding_time_sec"]].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig(out_dir / "correlation_heatmap.png", dpi=160)
            plt.close()

            # Query similarity vs decoding time
            plt.figure(figsize=(5, 4))
            sns.scatterplot(
                data=df,
                x="avg_decoding_time_sec",
                y="avg_query_sim",
                size="total_tokens",
                hue="p_init",
                palette="viridis",
            )
            plt.title("Time vs Query Sim")
            plt.tight_layout()
            plt.savefig(out_dir / "time_vs_query_sim.png", dpi=160)
            plt.close()
    else:
        # Fallback summary without pandas
        best_record = max(trial_records, key=lambda r: r["avg_query_sim"])
        with open(summary_json, "w") as f:
            json.dump(best_record, f, indent=2)
        print("Best config (by avg_query_sim):")
        print(json.dumps(best_record, indent=2))

    print("Search complete. Outputs stored in", args.out_dir)


if __name__ == "__main__":
    main()
