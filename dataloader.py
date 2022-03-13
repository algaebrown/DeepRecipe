import random
import numpy as np
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from file_utils import *
from vocab import *
# from glove_utils.glove import *


def get_recipe_dataloader(dataset, config_data, in_collate_fn):
    return DataLoader(dataset, batch_size=config_data['dataset']['batch_size'],
                      shuffle=True,
                      # num_workers=config_data['dataset']['num_workers'],
                      collate_fn=in_collate_fn,
                      # pin_memory=True
                      )

def find_img_path(imgid, root=IMAGES_PATH):
    return os.path.join(root, imgid[0], imgid[1], imgid[2], imgid[3], imgid)

def make_same_length(captions):
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)+2).long()

    #Tokenize
    for i, cap in enumerate(captions):
        targets[i, 0] = 1
        end = lengths[i]+1
        targets[i, 1:end] = cap[:end]
        targets[i,end] = 2

    return targets

def collate_fn(input_data):
    """Creates mini-batch tensors from the list of tuples (image, caption)
    by padding the captions to make them of equal length.
    We can not use default collate_fn because variable length tensors can't be stacked vertically.
    We need to pad the captions to make them of equal length so that they can be stacked for creating a mini-batch.
    Read this for more information - https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    Args:
        input_data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).

    # TODO: Do we really need to sort data?
    # input_data.sort(key=lambda x: len(x[3]), reverse=True)

    images, ing_binary, title, ing, ins, ann_id = zip(*input_data)

    images = torch.stack(images, 0)

    title = make_same_length(title)
    ing= make_same_length(ing)
    ins = make_same_length(ins)
    ing_binary = torch.stack(ing_binary, 0)
    
    return images, title, ing_binary, ing, ins, ann_id

def get_datasets(config_data, binary_ing=True):
    images_root_dir = ROOT_DIR
    img_size = config_data['dataset']['img_size']

    train_file_path = os.path.join(images_root_dir, config_data['dataset']['train_pickle'])
    val_file_path = os.path.join(images_root_dir, config_data['dataset']['val_pickle'])
    test_file_path = os.path.join(images_root_dir, config_data['dataset']['test_pickle'])

    vocab_threshold = config_data['dataset']['vocabulary_threshold']  # TODO
    vocab, ingd_vocab = load_vocab(train_file_path, vocab_threshold)
    
    train_data_loader = get_recipe_dataloader(RecipeDataset(images_root_dir, train_file_path, vocab, ingd_vocab, img_size, binary_indexing = binary_ing), config_data, in_collate_fn=collate_fn)
    test_data_loader = get_recipe_dataloader(RecipeDataset(images_root_dir, test_file_path, vocab, ingd_vocab, img_size, binary_indexing = binary_ing), config_data, in_collate_fn=collate_fn)
    val_data_loader = get_recipe_dataloader(RecipeDataset(images_root_dir, val_file_path, vocab, ingd_vocab, img_size, binary_indexing = binary_ing), config_data, in_collate_fn=collate_fn)

    return vocab,ingd_vocab, train_data_loader, val_data_loader, test_data_loader

class RecipeDataset(data.Dataset):
    """for torch.utils.data.DataLoader"""

    def __init__(self, root, pickle_path, vocab, ingd_vocab, img_size, transform=None, binary_indexing = False):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            root: image directory.
            vocab: vocabulary wrapper.
            transform: image transformations.
        """

        self.binary_indexing = binary_indexing

        with open(pickle_path, 'rb') as f:
            dictionary = pickle.load(f)
        
        self.root = root
        self.dict = dictionary
        self.ids = list(dictionary.keys())
        self.max_ingd = 16
        
        self.vocab = vocab
        self.ingd_vocab = ingd_vocab
        self.n_category = len(self.ingd_vocab) # total number of ingredients, plus unk, pad and...
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3774, 0.1051, -0.1764], std=[1.1593, 1.1756, 1.1958])  # TODO, might need to change
        ])

        self.resize = transforms.Compose(
            [transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(img_size)])

    def sentence_to_tensor(self, caption):
        """ given sentence, convert to tokens """

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.vocab('<start>')]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)

        return target

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        ing_index_tensor = None

        ann_id = self.ids[index]        
        title = self.dict[ann_id]['title'].lower()
        ingridients = [self.ingd_vocab(i) for i in self.dict[ann_id]['ingredient_list']]
        instructions = ' '.join([i['text'] for i in self.dict[ann_id]['instructions']]).lower()
        img_id = random.choice(self.dict[ann_id]['images'])['id']  # can have multiple images, choose 1

        if self.binary_indexing:
             ing_index_tensor = torch.zeros(self.n_category)
             ing_index_tensor[np.array(ingridients)] = 1
        
        path = find_img_path(img_id)
        image = Image.open(path).convert('RGB')
        image = self.resize(image)
        image = self.normalize(np.asarray(image))

        # Convert caption (string) to word ids.
        title_tensor = self.sentence_to_tensor(title)
        ing_tensor = torch.tensor(ingridients)
        ins_tensor = self.sentence_to_tensor(instructions)

        return image, ing_index_tensor, title_tensor, ing_tensor, ins_tensor, ann_id

    def __len__(self):
        return len(self.ids)

