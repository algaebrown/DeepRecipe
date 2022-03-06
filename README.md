# DeepRecipe

step1: download images from here `wget http://data.csail.mit.edu/im2recipe/recipe1M_images_val.tar`. This thing is 20GB, so takes quite a while.

step2: upload `dataset.tar.gz` to DeepRecipe/ and then decompress **UPDATE** there is a new pickle available with parsed ingredient list!!

step3: modify default.json, point the `val/` folder and the `pickles` correctly.

step4: run `try_dataloader.ipynb`


## Using GloVE vectors
- download: `wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip`
- preprocess using `process_glove_embedding.ipynb`
- visualize things using `visualize_ingredient_glove_vector.ipynb`
- the dataloader takes `glove.py` to use the gloVE vectors

## Models
Currently there are 2 models: naive (image -> ingredient categories); attention (image -> masked embedding `res_attn.py`)
Training is by `pretrain_resnet_tagger.py` and `pretrain_resnet_embed.py` which is similar to `experiment.py` and `model handler`.
They use slightly different dataloader options and collate function. It's not elegant :(

see `try_dataloader` to find explanations

