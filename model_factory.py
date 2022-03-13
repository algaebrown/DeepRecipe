from Models.Baseline import *
from Models.Baseline_ResNet_LSTM import *


def get_model(config_data, vocab, ing_vocab):
    # Get Parameters which might be required to build model
    model = None
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    nlayer = config_data['model']['num_layers']

    # TODO: Move this from a string to Enum    
    if config_data["mode"] == "pretrain-resnet-tagger":
        n_layers = 2 if "fc2" in config_data["experiment_name"] else 1
        model = Baseline(outdim=len(ing_vocab), n_lyr=n_layers)
    elif config_data["mode"] == "baseline-ResNet-LSTM":
        if config_data["model"]["pretrained_embed"]:
            model = Baseline_ResNet_LSTM(ing_vocab_size=len(ing_vocab), n_lyr=nlayer, 
                                         use_pretrain_embed = True, indg_vocab = ing_vocab)
        else:
            model = Baseline_ResNet_LSTM(ing_vocab_size=len(ing_vocab), n_lyr=nlayer)

    return model
