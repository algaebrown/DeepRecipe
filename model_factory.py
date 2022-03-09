from Models.Baseline import *
from Models.Baseline_ResNet_LSTM import *


def get_model(config_data, vocab):
    # Get Parameters which might be required to build model
    model = None
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    nlayer = config_data['model']['num_layers']

    # TODO: Move this from a string to Enum    
    if config_data["mode"] == "pretrain-resnet-tagger":
        n_layers = 2 if "fc2" in config_data["experiment_name"] else 1
        model = Baseline(outdim=3144, n_lyr=n_layers)
    elif config_data["mode"] == "Basline_ResNet_LSTM":
        model = Baseline_ResNet_LSTM(outdim=3144, n_lyr=n_layers)

    return model
