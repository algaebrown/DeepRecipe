# copied from model_factory, feel free to add lines to print shapes of each tensor to understand what happened
import torch
import torch.nn as nn
from torchvision import models

def set_parameter_requires_grad(model, feature_extracting):
    " https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class ResNet_ing_tagger(nn.Module):
    ''' ResNet encoder'''

    def __init__(self, outdim=256, dropout_p = 0.2, n_lyr = 1):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        # 
        # remove resnet fully connected (fc) layer
        old_output_size = self.resnet.fc.in_features #[64, 1000]
        self.resnet.fc = nn.Identity()
        # save the original dimension size
        # self.fc = nn.Linear(old_output_size, outdim)
        if n_lyr == 1:
            self.new_fc = nn.Sequential(nn.Dropout(p=dropout_p),nn.Linear(old_output_size, outdim))
        else:
            self.new_fc = nn.Sequential(nn.Dropout(p=dropout_p),nn.Linear(old_output_size, 1024), nn.Linear(1024, outdim))
        self.sigm = nn.Sigmoid()
        

    def forward(self, x, fine_tune = True):
        # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        if fine_tune:
            # only train the new fully connected layer
            with torch.no_grad():
                features = self.resnet(x)  # batchsize, 2048

        else:
            self.resnet.train()
            set_parameter_requires_grad(self.resnet, False)
            with torch.set_grad_enabled(mode = True):
                features = self.resnet(x)
                #print('using grad?', features.requires_grad)
        
        
        
        return self.sigm(self.new_fc(features))


class LSTM_decoder(nn.Module):
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
        self.dropout = nn.Dropout(p=0.2)
        
        self.model_type = model_type
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
        
        if self.model_type == 'LSTM':
            hidden = (weight.new(self.num_layer, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.num_layer, batch_size, self.hidden_dim).zero_())
        else:
            hidden = weight.new(self.num_layer, batch_size, self.hidden_dim).zero_()

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

        init_embed = self.dropout(self.graph_to_embedding(graph_embedding).unsqueeze(dim=1))  # [batch_size, 1, embed_size]
        captions = self.word_embeddings(captions)  # [batch_size, 18, embed_size]


        # concat embeds
        inputs = torch.cat((init_embed, captions), axis=1)
        # packed_input = pack_padded_sequence(inputs, sentence.shape[1], batch_first=True)

        lstm_out, hidden = self.rnn(inputs)

        lstm_feats = self.dropout(self.hidden_to_word(lstm_out))  # perform learn Wh+b
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


class Graph2Caption(nn.Module):
    def __init__(self, outdim=2048,
                 embedding_size=256,
                 hidden_size=256,
                 vocab_size=14463,
                 num_layer=2, model_type="LSTM"):
        super().__init__()

        self.encoder = ResNet_encoder(outdim=outdim)
        self.decoder = LSTM_decoder(graph_out_size=outdim,
                                    embedding_size=embedding_size,
                                    hidden_size=hidden_size,
                                    vocab_size=vocab_size,
                                    num_layer=num_layer, model_type=model_type)

    def forward(self, img, sentence):
        feat = self.encoder(img)
        pred = self.decoder(feat, sentence)

        return pred  # directly feed to output

    def generate_caption(self, img, size=22, mode='deterministic', gamma=1):
        ''' generate caption using different modes'''

        # generate feature first
        feat = self.encoder(img)

        words = self.decoder.predict(feat, size, mode=mode, r=gamma)

        return words


def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    nlayer = config_data['model']['num_layers']
    # You may add more parameters if you want
    
    if config_data["mode"] == "pretrain-resnet-tagger":
        if "fc2" in config_data["experiment_name"]:
            # more fully connected lyr
            model = ResNet_ing_tagger(outdim=3144, n_lyr=2)
            print('2 fully connected lyr')
        else:
            model = ResNet_ing_tagger(outdim = 3144)
    else:
        

        model = Graph2Caption(embedding_size=embedding_size,
                              hidden_size=hidden_size, 
                              num_layer=nlayer, 
                              vocab_size=len(vocab), 
                              model_type=model_type)

    # raise NotImplementedError("Model Factory Not Implemented")
    return model
