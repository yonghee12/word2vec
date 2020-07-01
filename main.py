from itertools import chain

import torch
import torch.nn.functional as F
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from torch.autograd import Variable

from functions import *

corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]


class BagOfWords:
    def __init__(self, list_of_texts=None):
        lemm = WordNetLemmatizer()
        self.lemmatizer = lemm.lemmatize

        if list_of_texts:
            self.make_bow(list_of_texts)

    def make_tokenized_matrix_english(self, texts: List[str], lemmatize=True):
        if lemmatize:
            self.tokenized_matrix = [[self.lemmatizer(word) for word in word_tokenize(text)] for text in texts]
        else:
            self.tokenized_matrix = [word_tokenize(text) for text in texts]

    def make_bow(self):
        assert self.tokenized_matrix
        self.unique_tokens = get_uniques_from_nested_lists(self.tokenized_matrix)
        self.token2idx, self.idx2token = get_item2idx(self.unique_tokens, unique=True)
        self.vocab_size = len(self.token2idx)

    def get_window_pairs(self, tokens: List[str], win_size=4, as_index=True) -> List[Tuple]:
        window_pairs = []
        for idx, token in enumerate(tokens):
            start = max(0, idx - win_size)
            end = min(len(tokens), idx + win_size + 1)
            for win_idx in range(start, end):
                if not idx == win_idx:
                    pair = (token, tokens[win_idx])
                    pair = pair if not as_index else tuple(self.token2idx[t] for t in pair)
                    window_pairs.append(pair)
        return window_pairs

    def make_pairs_matrix(self, win_size, as_index=True):
        self.pairs_matrix = [self.get_window_pairs(sent, win_size, as_index) for sent in self.tokenized_matrix]
        self.pairs_flat = list(chain.from_iterable(self.pairs_matrix))


class TrainWord2Vec:
    def __init__(self):
        pass

    def prepare_corpus(self, corpus):
        self.bow = BagOfWords()
        self.bow.make_tokenized_matrix_english(corpus, lemmatize=True)
        self.bow.make_bow()
        self.bow.make_pairs_matrix(win_size=2, as_index=True)
        self.vocab_size = len(self.bow.token2idx)
        self.train_size = len(self.bow.pairs_flat)

    def get_input_layer(self, word_idx):
        layer = torch.zeros(self.vocab_size)
        layer[word_idx] = 1.0
        return layer

    def train(self, emb_dimension=30, epochs=10, lr=0.001, continue_last=False):
        if not continue_last:
            self.W_c = Variable(torch.nn.init.xavier_normal(torch.empty(emb_dimension, self.vocab_size)),
                                requires_grad=True).float()
            self.W_o = Variable(torch.nn.init.xavier_normal(torch.empty(self.vocab_size, emb_dimension)),
                                requires_grad=True).float()

        for epoch in range(epochs):
            loss_val = 0
            for center_i, context_i in self.bow.pairs_flat:
                layer_input = Variable(self.get_input_layer(center_i)).float()
                y_true = Variable(torch.from_numpy(np.array([context_i])).long())

                layer_fc1 = torch.matmul(self.W_c, layer_input)
                layer_fc2 = torch.matmul(self.W_o, layer_fc1)
                layer_output = F.log_softmax(layer_fc2, dim=0)

                loss = F.nll_loss(layer_output.view(1, self.vocab_size), y_true)
                loss_val += loss.item()
                loss.backward()

                self.W_c.data -= lr * self.W_c.grad.data
                self.W_o.data -= lr * self.W_o.grad.data

                self.W_c.grad.data.zero_()
                self.W_o.grad.data.zero_()
            if epoch % 10 == 0:
                print(f"Loss at this epoch {epoch}: {loss_val / self.train_size}")


wv_trainer = TrainWord2Vec()
wv_trainer.prepare_corpus(corpus)
emb_dim = 5
wv_trainer.train(emb_dimension=emb_dim, epochs=1000, continue_last=False, lr=0.01)
wo_arr = np.array(wv_trainer.W_o.data.view(-1, emb_dim).data)
wo_df = pd.DataFrame(wo_arr, index=wv_trainer.bow.unique_tokens)
wc_arr = np.array(wv_trainer.W_c.data.T.data)
wc_df = pd.DataFrame(wc_arr, index=wv_trainer.bow.unique_tokens)


def get_cos_sim_score(wv, k1, k2):
    return round(cos_sim(wv.loc[k1, :], wv.loc[k2, :]), 3)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


get_cos_sim_score(wc_df, 'king', 'queen')

for perp in range(1, 5):
    tsne_plot(wv_trainer.bow.unique_tokens, wc_arr, filename=f'wc_word_vector.jpg', perplexity=perp)

print()

"""
인사이트
    - 지금으로선 wo보다 wc가 의미를 더 잘 파악하는 것 같다. 이유는 잘 모르겠다.
    - 적은 텍스트로도 의미를 어느정도 뽑아낸다.
    - normal보다 xavier가 조금 더 나은 것 같지만 절대적인지는 모르겠다. 어짜피 활성화함수가 없어서
    - optimizer는 시도를 해봐야겠다
    - 이렇게 무식하게 다 도는 방법 말고 없나?
        - skipgram 이니 한번에 context words 많이 아므로 그것 이용하는 방
TODO: optimizer, 한국어 토크나이징, most_similar 구현, plotting 자동화(perp 찾기), 좋은 dim 찾 
"""