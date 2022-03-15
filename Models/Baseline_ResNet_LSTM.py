# This model makes use of set of ingredients to predict the ingredients. I have to come up with a proper name 
# TODO: Not yet completed
from multiprocessing.pool import IMapIterator
import torch
import math
import torch.nn as nn

from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from glove_utils.glove import *
from transformer_utils import *

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


class ImageEncoder(nn.Module):
    def __init__(self, embed_size, dropout = 0.5):
        super(ImageEncoder, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        
        # remove resnet fully connected (fc) layer
        old_output_size = self.resnet.fc.in_features  # [64, 1000]
        self.resnet.fc = nn.Identity()

        self.new_fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(old_output_size, embed_size))

    def forward(self, image, fine_tune=False):
        if fine_tune:
            features = self.resnet(image) 
        else:
            with torch.no_grad():
                features = self.resnet(image)

        features = self.new_fc(features)
        return features


class IngredientDecoder(nn.Module):
    """Transformer decoder."""

    def __init__(self, embed_size, vocab_size, dropout=0.5, seq_length=20, num_instrs=15,
                 attention_nheads=16, pos_embeddings=True, num_layers=8, learned=True, normalize_before=True,
                 normalize_inputs=False, last_ln=False, scale_embed_grad=False):
        super(IngredientDecoder, self).__init__()
        self.dropout = dropout
        self.seq_length = seq_length * num_instrs

        # TODO: Check what the padding idx means
        self.embed_tokens = nn.Embedding(vocab_size, embed_size, padding_idx=vocab_size-1)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=embed_size ** -0.5)
        
        # Note: pos_embedding is false for the ingredient decoder
        if pos_embeddings:
            self.embed_positions = PositionalEmbedding(1024, embed_size, 0, left_pad=False, learned=learned)
        else:
            self.embed_positions = None

        # Normalize the inputs
        self.normalize_inputs = normalize_inputs
        if self.normalize_inputs:
            self.layer_norms_in = nn.ModuleList([LayerNorm(embed_size) for i in range(3)])

        self.embed_scale = math.sqrt(embed_size)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embed_size, attention_nheads, dropout=dropout, normalize_before=normalize_before,
                                    last_ln=last_ln)
            for i in range(num_layers)
        ])

        self.linear = Linear(embed_size, vocab_size-1)

    def forward(self, ingr_features, ingr_mask, captions, img_features, incremental_state=None):

        # NOTE: We wont be using ingr featues and its mask for the ingredient decoder
        # if ingr_features is not None:
        #     ingr_features = ingr_features.permute(0, 2, 1)
        #     ingr_features = ingr_features.transpose(0, 1)
        #     if self.normalize_inputs:
        #         self.layer_norms_in[0](ingr_features)

        if img_features is not None:
            img_features = img_features.permute(0, 2, 1)
            img_features = img_features.transpose(0, 1)
            if self.normalize_inputs:
                self.layer_norms_in[1](img_features)

        # if ingr_mask is not None:
        #     ingr_mask = (1-ingr_mask.squeeze(1)).byte()

        # NOTE: Psition embedding is not used for the Ingredient decoder
        # embed positions
        # if self.embed_positions is not None:
        #     positions = self.embed_positions(captions, incremental_state=incremental_state)


        # TODO: Incremental state mighe be always none
        if incremental_state is not None:
            if self.embed_positions is not None:
                positions = positions[:, -1:]
            captions = captions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(captions)

        if self.embed_positions is not None:
            x += positions

        if self.normalize_inputs:
            x = self.layer_norms_in[2](x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for p, layer in enumerate(self.layers):
            x  = layer(
                x,
                ingr_features,
                ingr_mask,
                incremental_state,
                img_features
            )
            
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        x = self.linear(x)
        _, predicted = x.max(dim=-1)

        return x, predicted

    # TODO: Update the first and last token value
    def sample(self, ingr_features, ingr_mask, greedy=True, temperature=1.0, beam=-1,
               img_features=None, first_token_value=0,
               replacement=True, last_token_value=0):

        incremental_state = {}

        # create dummy previous word
        if ingr_features is not None:
            fs = ingr_features.size(0)
        else:
            fs = img_features.size(0)

        # NOTE: I dont know what the beam means. Anyways for the ingredient decoder it is passed as -1
        # if beam != -1:
        #     if fs == 1:
        #         return self.sample_beam(ingr_features, ingr_mask, beam, img_features, first_token_value,
        #                                 replacement, last_token_value)
        #     else:
        #         print ("Beam Search can only be used with batch size of 1. Running greedy or temperature sampling...")

        first_word = torch.ones(fs)*first_token_value

        first_word = first_word.to(device).long()
        sampled_ids = [first_word]
        logits = []

        for i in range(self.seq_length):
            # forward
            outputs, _ = self.forward(ingr_features, ingr_mask, torch.stack(sampled_ids, 1),
                                      img_features, incremental_state)
            outputs = outputs.squeeze(1)
            if not replacement:
                # predicted mask
                if i == 0:
                    predicted_mask = torch.zeros(outputs.shape).float().to(device)
                else:
                    # ensure no repetitions in sampling if replacement==False
                    batch_ind = [j for j in range(fs) if sampled_ids[i][j] != 0]
                    sampled_ids_new = sampled_ids[i][batch_ind]
                    predicted_mask[batch_ind, sampled_ids_new] = float('-inf')

                # mask previously selected ids
                outputs += predicted_mask

            logits.append(outputs)
            if greedy:
                outputs_prob = torch.nn.functional.softmax(outputs, dim=-1)
                _, predicted = outputs_prob.max(1)
                predicted = predicted.detach()
            else:
                k = 10
                outputs_prob = torch.div(outputs.squeeze(1), temperature)
                outputs_prob = torch.nn.functional.softmax(outputs_prob, dim=-1).data

                # top k random sampling
                prob_prev_topk, indices = torch.topk(outputs_prob, k=k, dim=1)
                predicted = torch.multinomial(prob_prev_topk, 1).view(-1)
                predicted = torch.index_select(indices, dim=1, index=predicted)[:, 0].detach()

            sampled_ids.append(predicted)

        sampled_ids = torch.stack(sampled_ids[1:], 1)
        logits = torch.stack(logits, 1)

        return sampled_ids, logits

    def sample_beam(self, ingr_features, ingr_mask, beam=3, img_features=None, first_token_value=0,
                   replacement=True, last_token_value=0):
        k = beam
        alpha = 0.0
        # create dummy previous word
        if ingr_features is not None:
            fs = ingr_features.size(0)
        else:
            fs = img_features.size(0)
        first_word = torch.ones(fs)*first_token_value

        first_word = first_word.to(device).long()

        sequences = [[[first_word], 0, {}, False, 1]]
        finished = []

        for i in range(self.seq_length):
            # forward
            all_candidates = []
            for rem in range(len(sequences)):
                incremental = sequences[rem][2]
                outputs, _ = self.forward(ingr_features, ingr_mask, torch.stack(sequences[rem][0], 1),
                                          img_features, incremental)
                outputs = outputs.squeeze(1)
                if not replacement:
                    # predicted mask
                    if i == 0:
                        predicted_mask = torch.zeros(outputs.shape).float().to(device)
                    else:
                        # ensure no repetitions in sampling if replacement==False
                        batch_ind = [j for j in range(fs) if sequences[rem][0][i][j] != 0]
                        sampled_ids_new = sequences[rem][0][i][batch_ind]
                        predicted_mask[batch_ind, sampled_ids_new] = float('-inf')

                    # mask previously selected ids
                    outputs += predicted_mask

                outputs_prob = torch.nn.functional.log_softmax(outputs, dim=-1)
                probs, indices = torch.topk(outputs_prob, beam)
                # tokens is [batch x beam ] and every element is a list
                # score is [ batch x beam ] and every element is a scalar
                # incremental is [batch x beam ] and every element is a dict


                for bid in range(beam):
                    tokens = sequences[rem][0] + [indices[:, bid]]
                    score = sequences[rem][1] + probs[:, bid].squeeze().item()
                    if indices[:,bid].item() == last_token_value:
                        finished.append([tokens, score, None, True, sequences[rem][-1] + 1])
                    else:
                        all_candidates.append([tokens, score, incremental, False, sequences[rem][-1] + 1])

            # if all the top-k scoring beams have finished, we can return them
            ordered_all = sorted(all_candidates + finished, key=lambda tup: tup[1]/(np.power(tup[-1],alpha)),
                                 reverse=True)[:k]
            if all(el[-1] == True for el in ordered_all):
                all_candidates = []

            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1]/(np.power(tup[-1],alpha)), reverse=True)
            # select k best
            sequences = ordered[:k]
            finished = sorted(finished,  key=lambda tup: tup[1]/(np.power(tup[-1],alpha)), reverse=True)[:k]

        if len(finished) != 0:
            sampled_ids = torch.stack(finished[0][0][1:], 1)
            logits = finished[0][1]
        else:
            sampled_ids = torch.stack(sequences[0][0][1:], 1)
            logits = sequences[0][1]
        return sampled_ids, logits

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            if 'decoder.embed_positions._float_tensor' not in state_dict:
                state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor()
        return state_dict

    

        
class Baseline_ResNet_LSTM(nn.Module):
    """ ResNet encoder"""

    def __init__(self, embed_size, ing_vocab_size, dropout=0.5, num_layer=3, 
                 embedding_size=256, hidden_size=512, nhead=4, seq_length=30,
                 use_pretrain_embed = False):
        super().__init__()

        assert ing_vocab_size!=0
        
        self.image_encoder = ImageEncoder(embed_size=embed_size)


        self.ingr_decoder = IngredientDecoder(embed_size, ing_vocab_size, dropout=dropout,
                                      seq_length=seq_length,
                                      num_instrs=1, attention_nheads=nhead,
                                      pos_embeddings=False,
                                      num_layers=hidden_size,
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)



        #Encoder
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
        image = input['image']
        ingredients = input['ingredient']
        
        init_embed = self.graph_to_embedding(image).unsqueeze(dim=1)  # [batch_size, 1, embed_size]
        ingredients = self.word_embeddings(ingredients)  # [batch_size, 18, embed_size]

        # concat embeds
        inputs = torch.cat((init_embed, ingredients), axis=1)
        # lstm_out, hidden = self.ingredient_decoder(inputs)
        
        inputs = self.pos_encoder(inputs)
        output = self.transformer_encoder(inputs, mask = mask)

        self.lstm_feats = self.hidden_to_word(output)  # perform learn Wh+b
        
        decoder_out = nn.functional.softmax(self.lstm_feats[:, :-1, :], dim=2)  # the last prediction makes no sense
        ing_prob = torch.amax(decoder_out, dim=1)
        return ing_prob
    def sample(self, mode = 'stochastic', r = 0.9):
        if mode == 'deterministic':
            softmax_out = nn.functional.softmax(self.lstm_feats[:, :-1, :], dim=2) 
            words_index = torch.argmax(softmax_out, dim=2)

        else:
            # make weighted softmax
            weighted_out = self.lstm_feats / r

            softmax_out = nn.functional.softmax(weighted_out[:, :-1, :], dim=2) # 0: batch_size, 1: sentence len 2
            predicted_words = []
            # sample next word randomly
            for slen in range(softmax_out.shape[1]):
                slice_ = softmax_out[:, slen, :].squeeze(1)
                next_word = torch.multinomial(slice_, num_samples=1).squeeze(1)

                predicted_words.append(next_word)
            words_index = torch.stack(predicted_words, dim = 1)
        return words_index

    
    
    def predict(self, input,  mode = 'stochastic', r = 0.9):
        ''' generate text '''
        
        
        image = input['image']
        ingredients = input['ingredient']
        bptt = ingredients.shape[0]
        
        src_mask = generate_square_subsequent_mask(bptt).to('cuda:0')
        
        
        
            
        batch_size = ingredients.shape[0]
        if batch_size != bptt:
            src_mask = src_mask[:batch_size, :batch_size]
        ing_prob = self.forward(input, mask = src_mask)
        
        sampled_word = self.sample(mode = mode, r = r)
        
        
        return ing_prob, sampled_word
        
    def get_input_and_target_feature(self):
        ''' return a dictionary about what should be the input and what should be the output '''
        
        self.input_outputs = {'input': ['image', 'ingredient'],
             'output': ['ingredient_binary']
            }
        return self.input_outputs
    
    def get_loss_criteria(self):
        ''' return the loss class, and what should be the "label" '''
        return torch.nn.BCELoss()
