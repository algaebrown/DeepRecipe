##Taken from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py
import os
import pickle
from collections import Counter
from file_utils import ROOT_DIR

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
    ''' load both regular vocabulary and ingredient vocabs '''
    
    
    ####### regular vocabulary #############
    vocab_path=os.path.join(ROOT_DIR,'savedVocab')
    if os.path.isfile(vocab_path):
        with open(vocab_path, 'rb') as savedVocab:
            vocab = pickle.load(savedVocab)
            print("Using the saved vocab.")

    else:
        vocab = build_vocab(pickle_path, threshold)
        with open(vocab_path, 'wb') as savedVocab:
            pickle.dump(vocab, savedVocab)
            print("Saved the vocab.")
            
    ####### ingredient vocabulary #############
    ingd_vocab_path=os.path.join(ROOT_DIR, 'savedIngdVocab')
    if os.path.isfile(ingd_vocab_path):
        with open(ingd_vocab_path, 'rb') as savedVocab:
            ingd_vocab = pickle.load(savedVocab)
            print("Using the saved ingredient vocab.")

    else:
        ingd_vocab = build_ingredient_vocab(pickle_path, threshold)
        with open(ingd_vocab_path, 'wb') as savedVocab:
            pickle.dump(ingd_vocab, savedVocab)
            print("Saved the ingredient vocab.")

    return vocab, ingd_vocab


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

def build_ingredient_vocab(pickle_path, threshold):
    ''' build Vocab for list of ingredients '''
    with open(pickle_path, 'rb') as f:
        train_data = pickle.load(f)
    
    counter = Counter()
    ids = train_data.keys()
    for i, id_ in enumerate(ids):
        
        ingredients = train_data[id_]['ingredient_list']
        
                           
        counter.update(ingredients)
        

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the ingredients.".format(i + 1, len(ids)))

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
