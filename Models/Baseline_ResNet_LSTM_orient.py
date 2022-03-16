# This model makes use of set of ingredients to predict the ingredients. I have to come up with a proper name 
# TODO: Not yet completed
import torch
import math
import torch.nn as nn

from torchvision import models

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from glove_utils.glove import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mask_from_eos(ids, eos_value, mult_before=True):
    mask = torch.ones(ids.size()).to(device).byte()
    mask_aux = torch.ones(ids.size(0)).to(device).byte()

    # find eos in ingredient prediction
    for idx in range(ids.size(1)):
        # force mask to have 1s in the first position to avoid division by 0 when predictions start with eos
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
        else:
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
            mask[:, idx] = mask[:, idx] * mask_aux
    return mask

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

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

    def __init__(self, outdim=256, dropout=0.2, n_lyr=1, num_layer=3, 
                 embedding_size=256, hidden_size=512, ing_vocab_size=0 , nhead=4, 
                 use_pretrain_embed = False, indg_vocab = None):
        super().__init__()

        assert ing_vocab_size!=0
        
        print('orient no teacher')
        self.pad_value = 0
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

        
        if not use_pretrain_embed:
            self.word_embeddings = nn.Embedding(ing_vocab_size, embedding_size)
        else:
            print('initialize embed using GloVe pretrained vectors')
            self.word_embeddings, num_embeddings, embedding_dim = create_ing_emb_layer(indg_vocab, trainable=True)
            embedding_size = embedding_dim # override embed size
        
        self.graph_to_embedding = nn.Linear(old_output_size, embedding_size)

        # self.ingredient_decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=num_layer)
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layer)

        self.hidden_to_word = nn.Linear(embedding_size, ing_vocab_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input, mask = None, fine_tune=False):
        # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
        # I dont have to fine tune it
#         image = input['image']
#         ingredients = input['ingredient']
#         with torch.no_grad():
#             image = self.image_encoder(image)  # batchsize, 2048

#         init_embed = self.graph_to_embedding(image).unsqueeze(dim=1)  # [batch_size, 1, embed_size]
#         ingredients = self.word_embeddings(ingredients)  # [batch_size, 18, embed_size]

#         # concat embeds
#         inputs = torch.cat((init_embed, ingredients), axis=1) #[batch_size, 19, embed_size]
#         inputs = torch.permute(inputs, (1,0,2)) #[19, batch_size, embed_size]
#         # lstm_out, hidden = self.ingredient_decoder(inputs)
        
#         bptt = inputs.shape[0]
        
#         src_mask = generate_square_subsequent_mask(bptt).to('cuda:0')
#         print(f'mask_shape {src_mask.shape}')
        
        
#         inputs = self.pos_encoder(inputs)
#         output = self.transformer_encoder(inputs, mask = src_mask)

#         self.lstm_feats = torch.permute(self.hidden_to_word(output), (1,0,2))  # perform learn Wh+b # [batch_size, seq_len, ntoken]
        
        
#         decoder_out = nn.functional.softmax(self.lstm_feats[:, :-1, :], dim=2)  # the last prediction makes no sense
#         ing_prob = torch.amax(decoder_out, dim=1)

        # train with sampling
        ing_prob, word_idx, eos = self.predict(input, mode = 'deterministic', bptt = input['ingredient'].shape[1])
        return ing_prob, word_idx, eos
    
    
    def predict(self, input,  mode = 'stochastic', r = 0.9, bptt = 20):
        ''' generate text '''
        
        
        image = input['image']
        target_ingrs = input['ingredient']
        
        with torch.no_grad():
            image = self.image_encoder(image)  # batchsize, 2048

        img_embed = self.graph_to_embedding(image).unsqueeze(dim=1) # need to append to this
        
        pred_embed = [img_embed]
        pred_word_idx = []
        decoder_out = []
        
        for i in range(bptt):
            # stack along time
            if i == 0:
                inputs = img_embed
            else:
                inputs = torch.cat(pred_embed, dim = 1)
                
            
            # make it work as it should be
            inputs = torch.permute(inputs, (1,0,2))
            
            output = self.transformer_encoder(inputs)[-1, :, :].unsqueeze(0) # getting the last perdiction
            
            
            lstm_feats = torch.permute(self.hidden_to_word(output), (1,0,2)) # [batch_size, seq_len = 1, n_indg]
            
            if mode == 'deterministic':
                softmax_out = nn.functional.softmax(lstm_feats, dim=2).squeeze(1) # [batch_size, seq_len = 1, n_indg]
                
                word_index = torch.argmax(softmax_out, dim=1).unsqueeze(1)
                

            else:
                # make weighted softmax
                weighted_out = lstm_feats / r
                softmax_out = nn.functional.softmax(weighted_out, dim=2).squeeze(1) # # [batch_size, seq_len = 1, n_indg]
                
                word_index = torch.multinomial(softmax_out, num_samples=1) # [make batch_size, n_indg], output [batch_size]
                
            # make the word index into embedding
            #print('softmax out', softmax_out.shape)
            decoder_out.append(softmax_out)
            pred_word_idx.append(word_index)
            new_embed = self.word_embeddings(word_index) 

            pred_embed.append(new_embed)
        
        sampled_word = torch.cat(pred_word_idx, dim = 1)
        decoder_out = torch.stack(decoder_out, dim = 1) # stack [batch_size, n_indg] along dim 1 (seq_len)
        #print('decoder out', decoder_out.shape)

        eos = decoder_out[:, :, 2]
        
        # select transformer steps to pool from
        mask_perminv = mask_from_eos(target_ingrs, eos_value=2, mult_before=False)
        ingr_probs = decoder_out * mask_perminv.float().unsqueeze(-1)

        ingr_probs, _ = torch.max(ingr_probs, dim=1)
        sampled_word[mask_perminv == 0] = self.pad_value

            

        return ingr_probs, sampled_word, eos
        
    def get_input_and_target_feature(self):
        ''' return a dictionary about what should be the input and what should be the output '''
        
        self.input_outputs = {'input': ['image', 'ingredient'],
             'output': ['ingredient']
            }
        return self.input_outputs
    
    def get_loss_criteria(self, **kwargs):
        ''' return the loss class, and what should be the "label" '''
        return torch.nn.BCELoss(**kwargs)
