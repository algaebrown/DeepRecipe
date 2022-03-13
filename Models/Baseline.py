import torch
import torch.nn as nn

from torchvision import models
from model_utils import LossFeature


def set_parameter_requires_grad(model, feature_extracting):
    """ https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"""
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Baseline(nn.Module):
    """ ResNet encoder"""

    def __init__(self, outdim=256, dropout_p=0.2, n_lyr=1):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)

        # remove resnet fully connected (fc) layer
        old_output_size = self.resnet.fc.in_features  # [64, 1000]
        self.resnet.fc = nn.Identity()

        if n_lyr == 1:
            self.new_fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(old_output_size, outdim))
        else:
            self.new_fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(old_output_size, 1024), nn.Linear(1024, outdim))
            
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_dict,  title=None, ingredients=None, instructions=None, fine_tune=True):
        # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        
        x = input_dict['image']
        
        if fine_tune:
            with torch.no_grad():
                features = self.resnet(x)  # batchsize, 2048
        else:
            self.resnet.train()
            set_parameter_requires_grad(self.resnet, False)
            with torch.set_grad_enabled(mode=True):
                features = self.resnet(x)

        return self.sigmoid(self.new_fc(features))

    def get_input_and_target_feature(self):
        ''' return a dictionary about what should be the input and what should be the output '''
        
        self.input_outputs = {'input': ['image'],
             'output': ['ingredient_binary']
            }
        
        
        return self.input_outputs
    
    def get_loss_criteria(self):
        ''' return the loss class, and what should be the "label" '''
        return torch.nn.BCELoss()
