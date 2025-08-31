import math

import torch


def get_result_list_for_query(
    # Attack-specific:
    adv_passage_texts: List[str],
    query_id: str,
    # adv_passage_vecs: Union[List[torch.Tensor], torch.Tensor],  # TODO OPTIONAL
    # query_text: str,  # TODO to support custom queries we need to embed the entire corpus
    # General:
    model: RetrieverModel,
    queries: Dict[str, str],  # qid -> query text
    corpus: Dict[str, Dict[str, str]],  # pid -> passage text
    # To fetch `results`:
    dataset_name: str,
    data_split: str,
    data_portion: float,
    # results: Dict[str, Dict[str, float]] = None,
    # Metrics:
    top_k: int = 10,  # number of top passages to return
):
    # Load results (works for previously known queries)
    results = load_cached_eval(  # query_id -> {passage_id -> sim score}
        dataset_name=dataset_name,
        model_hf_name=model.model_hf_name,
        sim_func_name=model.sim_func_name,
        data_split=data_split,
        data_portion=data_portion,
    )
    if results is None:
        raise Exception("No cached retreival results found.")  # no cached results found

    # Get the query embeddings
    query_text = queries[query_id]
    query_emb = model.embed(texts=[query_text])

    # Calculate similarity scores for the adversarial passages
    adv_p_emb = model.embed(texts=adv_passage_texts)
    if model.sim_func_name == "cos_sim":
        adv_sim_scores = torch.nn.functional.cosine_similarity(query_emb, adv_p_emb)
    elif model.sim_func_name == "dot":
        adv_sim_scores = torch.matmul(query_emb, adv_p_emb.t())
    adv_sim_score = (
        adv_sim_scores.max().item()
    )  # we only care about the best passage (when examining a single query)
    adv_passage_text = adv_passage_texts[adv_sim_scores.argmax().item()]

    # Get the top-k passages for the query
    query_results = results[query_id]

    # calculate what's the rank of the adv passage:
    adv_rank = sum(1 for score in query_results.values() if score > adv_sim_score)
    adv_rank = math.inf if adv_rank == len(query_results) else adv_rank + 1

    # list top-k passages, including the adv-passage
    query_results["__adv__"] = (
        adv_sim_score  # include the adversarial passage in results
    )
    top_passages = sorted(query_results.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]
    top_passages_text = [
        corpus[pid]["text"] if pid != "__adv__" else adv_passage_text
        for pid, _ in top_passages
    ]

    return dict(
        query_text=query_text,
        adv_sim_score=adv_sim_score,
        adv_rank=adv_rank,
        top_passages=top_passages,
        top_passages_text=top_passages_text,
    )
