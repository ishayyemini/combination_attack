from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm


class BlackBoxAttack:
    def __init__(
        self,
        model: SentenceTransformer,
        q: str,
        # corpus: list[str],
        num_tokens=20,
        num_pool=500,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cos_sim=torch.nn.CosineSimilarity(dim=0, eps=1e-6),
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
        self.cos_sim = cos_sim.to(device)
        self.device = device

    def attack(self, p_adv: str):
        curr_p = p_adv
        tokens = []

        p_adv_emb = self.model.encode(p_adv, convert_to_tensor=True).to(self.device)
        print(f"initial similarity: {self.cos_sim(self.q_emb, p_adv_emb)}")

        curr_best = 0
        for n in range(self.num_tokens):
            print(f"iteration {n + 1}")
            best_token = None
            pool = np.random.choice(30521, size=(self.num_pool,))
            for i in tqdm(pool):
                token = self.model.tokenizer.decode(i)
                check_p = curr_p + " " + token
                check_emb = self.model.encode(check_p, convert_to_tensor=True)
                sim = self.cos_sim(self.q_emb, check_emb)
                if sim > curr_best:
                    best_token = token
                    curr_best = sim
            if best_token is not None:
                tokens.append(best_token)
                curr_p += " " + best_token
                print(f"best token: {best_token}, current similarity: {curr_best}")
                print()

        print(f"similarity with tokens: {curr_best}")

        return tokens

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
