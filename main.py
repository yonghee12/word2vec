from typing import *
from itertools import chain

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


class Word2Vec:
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
            self.W_c = Variable(torch.randn(emb_dimension, self.vocab_size), requires_grad=True).float()
            self.W_o = Variable(torch.randn(self.vocab_size, emb_dimension), requires_grad=True).float()

        for epoch in range(epochs):
            loss_val = 0
            for center_i, context_i in self.bow.pairs_flat:
                x = Variable(self.get_input_layer(center_i)).float()
                y_true = Variable(torch.from_numpy(np.array([context_i])).long())

                z1 = torch.matmul(self.W_c, x)
                z2 = torch.matmul(self.W_o, z1)

                log_softmax = F.log_softmax(z2, dim=0)
                loss = F.nll_loss(log_softmax.view(1, self.vocab_size), y_true)
                loss_val += loss.item()
                loss.backward()

                self.W_c.data -= lr * self.W_c.grad.data
                self.W_o.data -= lr * self.W_o.grad.data

                self.W_c.grad.data.zero_()
                self.W_o.grad.data.zero_()
            if epoch % 10 == 0:
                print(f"Loss at this epoch {epoch}: {loss_val / self.train_size}")


w2v = Word2Vec()
w2v.prepare_corpus(corpus)
w2v.train(emb_dimension=10, epochs=1000, continue_last=False, lr=0.01)
wo_arr = np.array(w2v.W_o.data.view(-1, 10).data)
wo_df = pd.DataFrame(wo_arr, index=w2v.bow.unique_tokens)
wc_arr = np.array(w2v.W_c.data.T.data)
wc_df = pd.DataFrame(wc_arr, index=w2v.bow.unique_tokens)

def get_cos_sim_score(wv, k1, k2):
    return round(cos_sim(wv.loc[k1, :], wv.loc[k2, :]), 3)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


get_cos_sim_score(wc_df, 'king', 'queen')

for perp in range(10):
    tsne_plot(w2v.bow.unique_tokens, wc_arr, filename=f'wc_word_vector.jpg', perplexity=perp)

print()