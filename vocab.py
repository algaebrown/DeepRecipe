##Taken from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py
import os
import pickle
from collections import Counter

import nltk


# A simple wrapper class for Vocabulary. No changes are required in this file
class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word.lower() in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word.lower()]

    def __len__(self):
        return len(self.word2idx)


def load_vocab(pickle_path, threshold):
    if os.path.isfile('savedVocab'):
        with open('savedVocab', 'rb') as savedVocab:
            vocab = pickle.load(savedVocab)
            print("Using the saved vocab.")

    else:
        vocab = build_vocab(pickle_path, threshold)
        with open('savedVocab', 'wb') as savedVocab:
            pickle.dump(vocab, savedVocab)
            print("Saved the vocab.")

    return vocab


def build_vocab(pickle_path, threshold):
    with open(pickle_path, 'rb') as f:
        train_data = pickle.load(f)
    counter = Counter()
    ids = train_data.keys()
    for i, id_ in enumerate(ids):
        title = train_data[id_]['title'].lower()
        ing = ' '.join([i['text'] for i in train_data[id_]['ingredients']]).lower()
        ins = ' '.join([i['text'] for i in train_data[id_]['instructions']]).lower()

        tokens = nltk.tokenize.word_tokenize(' '.join([title, ing, ins]))
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab
