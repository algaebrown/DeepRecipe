# This model makes use of set of ingredients to predict the ingredients. I have to come up with a proper name 
# TODO: Not yet completed
import torch
import torch.nn as nn

from torchvision import models
from model_utils import LossFeature


def set_parameter_requires_grad(model, feature_extracting):
    """ https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Baseline_ResNet_LSTM(nn.Module):
    """ ResNet encoder"""

    def __init__(self, outdim=256, dropout_p=0.2, n_lyr=1, num_layer=2, embedding_size=256, hidden_size=512, vocab_size=0 ):
        super().__init__()

        #Encoder
        self.resnet = models.resnet50(pretrained=True)
        # remove resnet fully connected (fc) layer
        old_output_size = self.resnet.fc.in_features  # [64, 1000]
        self.resnet.fc = nn.Identity()

        if n_lyr == 1:
            self.new_fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(old_output_size, outdim))
        else:
            self.new_fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(old_output_size, 1024), nn.Linear(1024, outdim))
    
        # Decoder
        self.num_layer = num_layer
        self.graph_to_embedding = nn.Linear(outdim, embedding_size)

        self.hidden_dim = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=num_layer)
        
        # to softmax
        self.hidden_to_word = nn.Linear(hidden_size, vocab_size)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x, title=None, ingredients=None, instructions=None, fine_tune=False):
        # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        pass
    
    def get_target_feature(self):
        return LossFeature.INGREDIENT_EMBEDDING
