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
