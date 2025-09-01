import re
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import trange
from sentence_transformers import util


class BlackBoxAttack:
    def __init__(
        self,
        model: SentenceTransformer,
        q: str,
        # corpus: list[str],
        num_tokens=20,
        num_pool=500,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=128,
    ):
        # The embedding model
        self.model = model
        # The query we optimize for
        self.q_emb = self.model.encode(q, convert_to_tensor=True).to(device)
        # The corpus
        # self.corpus = corpus
        # Embedding of corpus
        # self.corpus_emb = open("")
        # Amount of tokens to append to p_adv
        self.num_tokens = num_tokens
        # Amount of tokens to draw on each round
        self.num_pool = num_pool
        # Compute the embedding of each token and store the result in a matrix E
        self.tokenizer = model.tokenizer
        self.device = device
        self.batch_size = batch_size  # added

    def attack(self, p_adv: str):
        curr_p = p_adv
        tokens = []

        p_adv_emb = self.model.encode(p_adv, convert_to_tensor=True).to(self.device)
        base_sim = util.cos_sim(self.q_emb, p_adv_emb).item()
        print(f"initial similarity: {base_sim}")

        iter_best_score = 0
        valid_vocab_ids = self._get_valid_vocab_ids()

        for n in range(self.num_tokens):
            print(f"iteration {n + 1}")
            pool = np.random.choice(valid_vocab_ids, size=(self.num_pool,))

            # compute current baseline similarity for this iteration
            p_adv_emb = self.model.encode(curr_p, convert_to_tensor=True).to(
                self.device
            )
            iter_best_score = util.cos_sim(self.q_emb, p_adv_emb).item()
            best_token = None

            # evaluate candidates in parallel by batching
            for i in trange(0, len(pool), self.batch_size):
                batch_ids = [
                    pool[i] for i in range(i, min(i + self.batch_size, len(pool)))
                ]
                batch_tokens = [self.model.tokenizer.decode(i) for i in batch_ids]

                # build candidate prompts
                check_ps = [curr_p + " " + t for t in batch_tokens]

                # encode all candidates as a single batch
                embs = self.model.encode(
                    check_ps,
                    convert_to_tensor=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                ).to(self.device)

                # vectorized cosine similarity against q_emb
                scores = util.cos_sim(self.q_emb, embs).squeeze(0)  # shape: [batch]
                max_score, max_idx = torch.max(scores, dim=0)
                ms = max_score.item()
                if ms > iter_best_score:
                    iter_best_score = ms
                    best_token = batch_tokens[max_idx.item()]

            if best_token is not None:
                tokens.append(best_token)
                curr_p += " " + best_token
                print(
                    f"best token: {best_token}, current similarity: {iter_best_score}\n"
                )
            else:
                print("no improving token found\n")

        print(f"final similarity: {iter_best_score}")
        return tokens

    def square_attack(
        self,
        base_prompt: str,
        total_tokens: int = 20,
        p_init: float = 0.5,
        num_iters: int = 500,
        random_pool_per_pos: int = 50,
        early_stop_patience: int = 100,
        seed: int | None = None,
        initial_tokens: list[str] | None = None,
    ):
        """A 1D adaptation of the image Square Attack for token sequence (prompt) optimization.

        Instead of square patches over image pixels, we sample contiguous blocks ("windows")
        over the sequence of appended tokens and refresh the entire block with randomly
        sampled vocabulary tokens. A proposal is accepted if it increases cosine similarity.

        Args:
            base_prompt: The initial (fixed) part of the prompt we optimize around.
            total_tokens: Number of optimizable tokens appended to base_prompt.
            p_init: Initial fraction of tokens replaced in one update (analogous to image pixel fraction).
            num_iters: Maximum number of iterations.
            random_pool_per_pos: For every position in the chosen block we sample uniformly that many candidate tokens and pick 1 (simple random draw). Higher => more diversity.
            early_stop_patience: Stop if no improvement for this many iterations.
            seed: Optional RNG seed for reproducibility.
            initial_tokens: Optional list of tokens to initialize the appended tokens (if shorter than total_tokens, repeated as needed).

        Returns:
            appended_tokens: Final list of appended tokens (length = total_tokens).
            final_prompt: The resulting full prompt string.
            history: List of (iteration, best_similarity) tracking progress (iteration 0 is initialization).
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        valid_vocab_ids = self._get_valid_vocab_ids()
        if total_tokens <= 0:
            raise ValueError("total_tokens must be > 0")

        # Initialize appended tokens randomly.
        appended_tokens = []
        if initial_tokens:
            appended_tokens = list(initial_tokens)

        for i in range(len(appended_tokens), total_tokens):
            if initial_tokens and len(initial_tokens) != 0:
                appended_tokens.append(initial_tokens[i % len(initial_tokens)])
            else:
                tok_id = np.random.choice(valid_vocab_ids)
                tok = self.tokenizer.decode(int(tok_id))
                if not tok.strip():  # ensure non-empty
                    tok = "the"  # fallback harmless common token
                appended_tokens.append(tok)

        def build_prompt(tokens_list):
            if tokens_list:
                return base_prompt + " " + " ".join(tokens_list)
            return base_prompt

        current_prompt = build_prompt(appended_tokens)
        with torch.no_grad():
            emb = self.model.encode(
                current_prompt, convert_to_tensor=True, show_progress_bar=False
            ).to(self.device)
            best_sim = util.cos_sim(self.q_emb, emb).item()

        history = [(0, best_sim)]
        best_tokens = list(appended_tokens)
        no_improve = 0

        for it in range(1, num_iters + 1):
            # Schedule analogous to square attack p-schedule.
            p = self._p_selection(p_init, it - 1, num_iters)
            block_size = max(1, int(round(p * total_tokens)))
            block_size = min(block_size, total_tokens)

            start = np.random.randint(0, total_tokens - block_size + 1)
            end = start + block_size

            proposal_tokens = list(best_tokens)
            # Refresh contiguous block.
            for pos in range(start, end):
                # sample a random candidate token (optionally from a small pool -> pick one randomly)
                # For simplicity we just sample one token; could expand to pool scoring if needed.
                pool_ids = np.random.choice(
                    valid_vocab_ids, size=(random_pool_per_pos,)
                )
                chosen_id = np.random.choice(pool_ids)
                new_tok = self.tokenizer.decode(int(chosen_id))
                if not new_tok.strip():
                    continue  # skip empty; retain old token
                proposal_tokens[pos] = new_tok

            proposal_prompt = build_prompt(proposal_tokens)
            with torch.no_grad():
                prop_emb = self.model.encode(
                    proposal_prompt,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                ).to(self.device)
                sim = util.cos_sim(self.q_emb, prop_emb).item()

            if sim > best_sim:
                best_sim = sim
                best_tokens = proposal_tokens
                current_prompt = proposal_prompt
                no_improve = 0
            else:
                no_improve += 1

            history.append((it, best_sim))

            # Early stopping
            if early_stop_patience and no_improve >= early_stop_patience:
                break

        return best_tokens, current_prompt, history

    def adversarial_decoding_rag(
        self,
        base_prompt: str,
        reference_texts: list[str],
        max_tokens: int = 32,
        beam_size: int = 5,
        pool_size: int = 120,
        seed: int | None = None,
        use_query_similarity: bool = True,
        sample_per_beam: int = 40,
        temperature: float = 1.0,
        verbose: bool = False,
    ):
        """Adversarial Decoding (RAG-oriented) to craft appended tokens that maximize
        semantic similarity to a target (query embedding and/or reference corpus embeddings).

        This is a lightweight, self-contained adaptation of the retrieval-focused
        adversarial decoding logic: we run a constrained beam search over appended
        tokens. At each expansion step we sample a candidate token pool (per beam),
        score all expansions via cosine similarity vs either:
          - the stored query embedding (self.q_emb) if use_query_similarity=True
          - the mean of reference_texts embeddings (always included)
        and keep the top `beam_size` sequences.

        Args:
            base_prompt: Fixed starting prompt (not altered).
            reference_texts: Texts whose embedding(s) guide optimization.
            max_tokens: Max number of appended tokens.
            beam_size: Beam width.
            pool_size: Size of global random vocab pool drawn once per step (upper bound for diversity).
            seed: RNG seed.
            use_query_similarity: Include similarity to original query embedding.
            sample_per_beam: For every beam we draw this many token candidates from the (possibly temperature-weighted) pool.
            temperature: Softmax temperature over frequency-derived scores (currently uniform if no freq stats available; kept for extensibility).
            verbose: Print progress if True.

        Returns:
            best_tokens: List[str] best appended tokens.
            final_prompt: The concatenated full prompt.
            history: List of dicts with iteration stats.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if not reference_texts:
            raise ValueError("reference_texts must be non-empty")

        # Encode reference texts once (batched)
        ref_embs = self.model.encode(
            reference_texts,
            convert_to_tensor=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).to(self.device)
        mean_ref = ref_embs.mean(dim=0, keepdim=True)  # [1, d]

        # Precompute base embedding (without any appended tokens)
        base_emb = self.model.encode(
            base_prompt,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).to(self.device)

        valid_vocab_ids = self._get_valid_vocab_ids()
        if len(valid_vocab_ids) == 0:
            raise RuntimeError("No valid vocab ids found.")

        # Beam entries: (tokens_list, score, embedding_cache_string)
        # We store only the appended tokens; full prompt constructed on demand.
        # Initial score is similarity of base prompt.
        def score_full_prompt(tokens_list: list[str]):
            full_text = (
                base_prompt
                if not tokens_list
                else base_prompt + " " + " ".join(tokens_list)
            )
            emb = self.model.encode(
                full_text,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).to(self.device)
            # Score parts
            scores = []
            # Always include mean reference similarity
            scores.append(util.cos_sim(emb, mean_ref).item())
            if use_query_similarity:
                scores.append(util.cos_sim(emb, self.q_emb).item())
            return sum(scores) / len(scores), emb

        base_score, _ = score_full_prompt([])
        beams = [([], base_score)]
        best_overall = ([], base_score)
        history = [{"step": 0, "best_score": base_score, "beam_scores": [base_score]}]

        for step in range(1, max_tokens + 1):
            # Sample a global pool of candidate vocab ids (without replacement if possible)
            if pool_size >= len(valid_vocab_ids):
                global_pool = valid_vocab_ids
            else:
                global_pool = np.random.choice(
                    valid_vocab_ids, size=pool_size, replace=False
                )

            # Token probabilities (currently uniform). Temperature kept for future frequency weighting.
            probs = np.ones(len(global_pool), dtype=np.float64)
            probs /= probs.sum()
            if temperature != 1.0:
                # Softmax with temperature (still uniform here)
                probs = np.exp(np.log(probs + 1e-12) / temperature)
                probs /= probs.sum()

            candidate_expansions = []  # list of (tokens_list, score)
            # Expand each beam
            for tokens_list, _score in beams:
                # Per-beam sample
                sample_ids = np.random.choice(
                    global_pool,
                    size=min(sample_per_beam, len(global_pool)),
                    replace=False,
                    p=probs,
                )
                decoded = [self.tokenizer.decode(int(i)) for i in sample_ids]
                # Build prompts in batch
                batch_prompts = []
                new_token_lists = []
                for tok in decoded:
                    if not tok.strip():
                        continue
                    new_tokens = tokens_list + [tok]
                    full_text = base_prompt + " " + " ".join(new_tokens)
                    batch_prompts.append(full_text)
                    new_token_lists.append(new_tokens)
                if not batch_prompts:
                    continue
                embs = self.model.encode(
                    batch_prompts,
                    convert_to_tensor=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ).to(self.device)
                # Compute similarity metrics
                ref_sims = util.cos_sim(embs, mean_ref.repeat(embs.size(0), 1)).squeeze(
                    1
                )
                if use_query_similarity:
                    q_sims = util.cos_sim(
                        embs, self.q_emb.repeat(embs.size(0), 1)
                    ).squeeze(1)
                    combined = (ref_sims + q_sims) / 2.0
                else:
                    combined = ref_sims
                for new_tokens, sc in zip(new_token_lists, combined.tolist()):
                    candidate_expansions.append((new_tokens, sc))

            if not candidate_expansions:
                # Cannot expand further
                break

            # Select top beams
            candidate_expansions.sort(key=lambda x: x[1], reverse=True)
            beams = candidate_expansions[:beam_size]

            # Update best
            if beams[0][1] > best_overall[1]:
                best_overall = beams[0]

            history.append(
                {
                    "step": step,
                    "best_score": best_overall[1],
                    "beam_scores": [b for (_, b) in beams],
                }
            )
            if verbose:
                print(
                    f"[RAG-ADV] step={step} best={best_overall[1]:.4f} top_beam={beams[0][1]:.4f}"
                )

        best_tokens, best_score = best_overall
        final_prompt = base_prompt + (
            " " + " ".join(best_tokens) if best_tokens else ""
        )
        return best_tokens, final_prompt, history

    def _p_selection(self, p_init: float, it: int, n_iters: int) -> float:
        """Piece-wise constant schedule for p (re-used from original Square Attack)."""
        scaled = int(it / n_iters * 10000)
        # Mirrors original schedule thresholds.
        if 10 < scaled <= 50:
            return p_init / 2
        elif 50 < scaled <= 200:
            return p_init / 4
        elif 200 < scaled <= 500:
            return p_init / 8
        elif 500 < scaled <= 1000:
            return p_init / 16
        elif 1000 < scaled <= 2000:
            return p_init / 32
        elif 2000 < scaled <= 4000:
            return p_init / 64
        elif 4000 < scaled <= 6000:
            return p_init / 128
        elif 6000 < scaled <= 8000:
            return p_init / 256
        elif 8000 < scaled <= 10000:
            return p_init / 512
        else:
            return p_init

    def _get_valid_vocab_ids(self):
        if hasattr(self, "_valid_vocab_ids"):
            return self._valid_vocab_ids

        vocab = self.tokenizer.get_vocab()
        valid_ids = []
        english_pattern = re.compile(r"^[a-zA-Z]+$")

        for token, token_id in vocab.items():
            if english_pattern.match(token):
                valid_ids.append(token_id)

        self._valid_vocab_ids = np.array(valid_ids)
        return self._valid_vocab_ids

    # def add_to_corpus(self, c: str):
    #     corpus = self.corpus.tolist()
    #     corpus.append(c)
    #     self.corpus = np.array(corpus)
    #     self.corpus_emb.append(self.model.encode(c, convert_to_tensor=True))

    # def get_top_k(self, q: str, k=5):
    #     q_emb = self.model.encode(q, convert_to_tensor=True)
    #     topk = torch.topk(
    #         torch.tensor([self.cos_sim(q_emb, c_emb) for c_emb in self.corpus_emb]), k=k
    #     )
    #     print(topk)
    #     return self.corpus[topk.indices]
