from train_ import *

eng_corpus = ['he is a king',
              'she is a queen',
              'he is a man',
              'she is a woman',
              'warsaw is poland capital',
              'berlin is germany capital',
              'paris is france capital']

wv_trainer = TrainWord2Vec()
wv_trainer.prepare_corpus(eng_corpus, win_size=2)

emb_dim = 5
wv_trainer.train(emb_dimension=emb_dim, epochs=10, continue_last=False, lr=0.01, verbose=0)
wv_trainer.train(emb_dimension=emb_dim, epochs=1000, continue_last=True, lr=0.01)
wv_trainer.train(emb_dimension=emb_dim, epochs=1000, continue_last=True, lr=0.005)

wo_arr = np.array(wv_trainer.context_vectors.data.view(-1, emb_dim).data)
wo_df = pd.DataFrame(wo_arr, index=wv_trainer.bow.unique_tokens)
wc_arr = np.array(wv_trainer.center_vectors.data.T.data)
wc_df = pd.DataFrame(wc_arr, index=wv_trainer.bow.unique_tokens)


def get_cos_sim_score(wv, k1, k2):
    return round(cos_sim(wv.loc[k1, :], wv.loc[k2, :]), 3)


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


get_cos_sim_score(wc_df, 'king', 'berlin')

for perp in range(1, 5):
    tsne_plot(wv_trainer.bow.unique_tokens, wc_arr, filename=f'remote_wc_word_vector.jpg', perplexity=4)
