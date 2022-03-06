from Models.Baseline import *
from model_utils import *



def get_model(model_name: ModelName, config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    nlayer = config_data['model']['num_layers']
    # You may add more parameters if you want
    if model_name == ModelName.BASELINE:
        model = Baseline(embedding_size=embedding_size,
                         hidden_size=hidden_size,
                         num_layer=nlayer,
                         vocab_size=len(vocab),
                         model_type=model_type)

    # We need to other else if statements as we create more

    # raise NotImplementedError("Model Factory Not Implemented")
    return model
