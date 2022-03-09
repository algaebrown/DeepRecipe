# derived from https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
# create embedding layer from pretrained GloVE 50

import pickle

# load preprocessed glove dataset
import numpy as np
import torch.nn as nn

from file_utils import GLOVE_PATH

npzfile = np.load(f'{GLOVE_PATH}/6B.50_words.npz')
vectors = npzfile['arr_0']
# vectors.flush()
words = pickle.load(open(f'{GLOVE_PATH}/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{GLOVE_PATH}/6B.50_idx.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}


def ingredient2embedding(ingd, emb_dim=50):
    ''' convert words of ingredient to embedding 
    wheat flour -> [1,0,0.5,1,2,0]
    '''
    sum_embedding = np.zeros(emb_dim)
    for word in ingd:
        try:
            sum_embedding += glove[word]
        except:
            pass
    return sum_embedding


def vocab2matrix(vocab, emd_dim=50):
    ''' fetch each vocabulary's embedding to matrix 
    vocab: vocab.Vocabulary object
    '''

    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0

    for idx in vocab:
        word = vocab.idx2word[idx]
        try:
            weights_matrix[idx] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[idx] = np.random.normal(scale=0.6, size=(emb_dim,))
    print(f'found {words_found} words')
    return weights_matrix


def create_emb_layer(vocab, trainable=True):
    weights_matrix = vocab2matrix(vocab)

    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)  # vocab (our own) index -> glove vector
    emb_layer.load_state_dict({'weight': weights_matrix})
    if True:
        emb_layer.weight.requires_grad = True

    return emb_layer, num_embeddings, embedding_dim
