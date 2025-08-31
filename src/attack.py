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
