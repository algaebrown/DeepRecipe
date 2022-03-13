# This model makes use of set of ingredients to predict the ingredients. I have to come up with a proper name 
# TODO: Not yet completed
import torch
import math
import torch.nn as nn

from torchvision import models
from model_utils import LossFeature
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def set_parameter_requires_grad(model, feature_extracting):
    """ https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Baseline_ResNet_LSTM(nn.Module):
    """ ResNet encoder"""

    def __init__(self, outdim=256, dropout=0.2, n_lyr=1, num_layer=3, embedding_size=256, hidden_size=512, ing_vocab_size=0 , nhead=4):
        super().__init__()

        assert ing_vocab_size!=0
        
        #Encoder
        self.image_encoder = models.resnet50(pretrained=True)
        
        # remove resnet fully connected (fc) layer
        old_output_size = self.image_encoder.fc.in_features  # [64, 1000]
        self.image_encoder.fc = nn.Identity()

        # if n_lyr == 1:
        #     self.new_fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(old_output_size, outdim))
        # else:
        #     self.new_fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(old_output_size, 1024), nn.Linear(1024, outdim))
    
        # Decoder
        self.num_layer = num_layer
        self.hidden_dim = hidden_size

        self.graph_to_embedding = nn.Linear(old_output_size, embedding_size)
        self.word_embeddings = nn.Embedding(ing_vocab_size, embedding_size)

        # self.ingredient_decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=num_layer)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)

        self.hidden_to_word = nn.Linear(embedding_size, ing_vocab_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, fine_tune=False):
        # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        # I dont have to fine tune it
        image = input['image']
        ingredients = input['ingredient']
        with torch.no_grad():
            image = self.image_encoder(image)  # batchsize, 2048

        init_embed = self.graph_to_embedding(image).unsqueeze(dim=1)  # [batch_size, 1, embed_size]
        ingredients = self.word_embeddings(ingredients)  # [batch_size, 18, embed_size]

        # concat embeds
        inputs = torch.cat((init_embed, ingredients), axis=1)
        # lstm_out, hidden = self.ingredient_decoder(inputs)
        
        inputs = self.pos_encoder(inputs)
        output = self.transformer_encoder(inputs)

        lstm_feats = self.hidden_to_word(output)  # perform learn Wh+b
        decoder_out = nn.functional.softmax(lstm_feats[:, :-1, :], dim=2)  # the last prediction makes no sense
        ing_prob = torch.amax(decoder_out, dim=1)
        return ing_prob

    def get_input_and_target_feature(self):
        ''' return a dictionary about what should be the input and what should be the output '''
        
        self.input_outputs = {'input': ['image', 'ingredient'],
             'output': ['ingredient_binary']
            }
        return self.input_outputs
    
    def get_loss_criteria(self):
        ''' return the loss class, and what should be the "label" '''
        return torch.nn.BCELoss()
