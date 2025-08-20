from sentence_transformers import SentenceTransformer
import torch
import numpy as np


class BlackBoxAttack:
    def __init__(
        self,
        model: SentenceTransformer,
        q: str,
        corpus: list[str],
        num_tokens=20,
        num_pool=500,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        # The embedding model
        self.model = model
        # The query we optimize for
        self.q_emb = self.model.encode(q, convert_to_tensor=True)
        # The corpus
        self.corpus = np.array(corpus)
        # Embedding of corpus
        self.corpus_emb = [self.model.encode(c, convert_to_tensor=True) for c in corpus]
        # Amount of tokens to append to p_adv
        self.num_tokens = num_tokens
        # Amount of tokens to draw on each round
        self.num_pool = num_pool
        # Compute the embedding of each token and store the result in a matrix E
        self.tokenizer = model.tokenizer
        self.cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def attack(self, p_adv: str):
        curr_p = p_adv
        tokens = []

        p_adv_emb = self.model.encode(p_adv, convert_to_tensor=True)
        print(f"initial similarity: {self.cos_sim(self.q_emb, p_adv_emb)}")

        curr_best = 0
        for n in range(self.num_tokens):
            print(f"iteration {n}")
            best_token = None
            for i in np.random.choice(30521, size=(self.num_pool,)):
                token = self.model.tokenizer.decode(i)
                check_p = curr_p + " " + token
                check_emb = self.model.encode(check_p, convert_to_tensor=True)
                sim = self.cos_sim(self.q_emb, check_emb)
                if sim > curr_best:
                    best_token = token
                    curr_best = sim
                    print(token)
            if best_token is not None:
                tokens.append(best_token)
                curr_p += " " + best_token

        print(f"similarity with tokens: {curr_best}")

        return tokens

    def add_to_corpus(self, c: str):
        corpus = self.corpus.tolist()
        corpus.append(c)
        self.corpus = np.array(corpus)
        self.corpus_emb.append(self.model.encode(c, convert_to_tensor=True))

    def get_top_k(self, q: str, k=5):
        q_emb = self.model.encode(q, convert_to_tensor=True)
        topk = torch.topk(
            torch.tensor([self.cos_sim(q_emb, c_emb) for c_emb in self.corpus_emb]), k=k
        )
        print(topk)
        return self.corpus[topk.indices]
