from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

# See this for input references - https://www.nltk.org/api/nltk.translate.html#nltk.translate.bleu_score.sentence_bleu
# A Caption should be a list of strings.
# Reference Captions are list of actual captions - list(list(str))
# Predicted Caption is the string caption based on your model's output - list(str)
# Make sure to process your captions before evaluating bleu scores -
# Converting to lower case, Removing tokens like <start>, <end>, padding etc.

def bleu1(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)


def bleu4(reference_captions, predicted_caption):
    return 100 * sentence_bleu(reference_captions, predicted_caption,
                               weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1)

def to_word(word_index, vocab):
    ''' convert word index to "words", [1,523,254,2] => 'Hi', 'there'
    remove weird tokens for u
    '''
    words = [vocab.idx2word[w].lower() for w in word_index]
    try:
        start_idx = words.index('<start>')+1
    except:
        start_idx = 0
    try:
        end_idx = words.index('<end>')-1
    except:
        end_idx = len(words)
    
    # remove token
    words = words[start_idx:end_idx]
    words = [w for w in words if w not in ['<pad>','<unk>']]
    
    return words

def get_all_captions(img_id, coco_test):
    ''' given img_id, return all captions '''
    return [[w.lower() for w in i['caption'].split(' ')] for i in coco_test.imgToAnns[img_id]]
    

# TODO: Need to develop the score method for ingredients list   
def compute_blue_score(img_ids, pred_words, vocab, coco_test, verbose = False):
    
    ''' return a list of BLEU1 score and BLEU4 score for a batch of data '''
    pass
