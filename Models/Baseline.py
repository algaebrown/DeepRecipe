import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ImageEncoder(nn.Module):
    ''' ResNet encoder'''

    def __init__(self, outdim=256):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        # remove resnet fully connected (fc) layer
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        with torch.no_grad():
            features = self.resnet(x)  # batchsize, 2048
        return features


# TODO: This needs to be changed according to our need to generate Ingridients with Transformer
class IngridientDecoder(nn.Module):
    def __init__(self, graph_out_size=2048,
                 embedding_size=256,
                 hidden_size=32,
                 vocab_size=14463,
                 num_layer=2, model_type="LSTM"):
        super().__init__()
        self.num_layer = num_layer
        self.graph_to_embedding = nn.Linear(graph_out_size, embedding_size)

        self.hidden_dim = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        if model_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=num_layer)
        else:
            self.rnn = nn.RNN(embedding_size, hidden_size, nonlinearity="relu", batch_first=True, num_layers=num_layer)

        # to softmax
        self.hidden_to_word = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.num_layer, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.num_layer, batch_size, self.hidden_dim).zero_())

        return hidden

    def forward(self, graph_embedding, sentence):
        '''
        graph_embedding: batch_size * decoder.outdim
        sentence: batch_size * 18 (length of sentence)
        
        '''

        # captions, caption_lengths = pad_packed_sequence(sentence, batch_first=True)
        # for i in range(captions.shape[0]):
        #     captions[i, caption_lengths[i]-1] = 0
        captions = sentence
                

        # e [layer, batch_size, hidden_size]

        init_embed = self.graph_to_embedding(graph_embedding).unsqueeze(dim=1)  # [batch_size, 1, embed_size]
        captions = self.word_embeddings(captions)  # [batch_size, 18, embed_size]


        # concat embeds
        inputs = torch.cat((init_embed, captions), axis=1)
        # packed_input = pack_padded_sequence(inputs, sentence.shape[1], batch_first=True)

        lstm_out, hidden = self.rnn(inputs)

        lstm_feats = self.hidden_to_word(lstm_out)  # perform learn Wh+b
        # decoder_out = nn.functional.softmax(lstm_feats[:, :-1, :], dim=2)  # the last prediction makes no sense
        return lstm_feats[:, :-1, :]  # [batch_size, sentence+1, vocab_size]


    def predict(self, graph_embedding, size, mode='deterministic', r=0.9):
        ''' given 1 word embedding and hidden, output the next word softmax
        mode: deterministic or stochiastic'''
        embed = self.graph_to_embedding(graph_embedding).unsqueeze(dim=1)
        hidden = self.init_hidden(graph_embedding.size()[0])
        predicted_words = []

        for i in range(size):
            lstm_out, hidden = self.rnn(embed, hidden)
            lstm_feats = self.hidden_to_word(lstm_out)

            if mode == 'deterministic':
                softmax_out = nn.functional.softmax(lstm_feats, dim=2)
                next_word = torch.argmax(softmax_out, dim=2)

            else:
                # make weighted softmax
                weighted_out = lstm_feats / r

                softmax_out = nn.functional.softmax(weighted_out, dim=2)

                # sample next word randomly
                next_word = torch.multinomial(softmax_out.squeeze(dim=1), num_samples=1)

            predicted_words.append(next_word)
            embed = self.word_embeddings(next_word)

        return torch.cat(predicted_words, dim=1)


class Baseline(nn.Module):
    def __init__(self, outdim=2048,
                 embedding_size=256,
                 hidden_size=256,
                 vocab_size=14463,
                 num_layer=2, model_type="LSTM"):
        super().__init__()

        self.image_encoder = ImageEncoder(outdim=outdim)
        self.ingredient_decoder = IngridientDecoder(graph_out_size=outdim,
                                    embedding_size=embedding_size,
                                    hidden_size=hidden_size,
                                    vocab_size=vocab_size,
                                    num_layer=num_layer, model_type=model_type)

    def forward(self, img, sentence):
        feat = self.image_encoder(img)
        pred = self.ingredient_decoder(feat, sentence)

        return pred  # directly feed to output

    def generate_ingredients_list(self, img, size=22, mode='deterministic', gamma=1):
        ''' generate caption using different modes'''

        # generate feature first
        feat = self.image_encoder(img)

        words = self.ingredient_decoder.predict(feat, size, mode=mode, r=gamma)

        return words
