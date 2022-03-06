import torch.utils.data as data
import torch
import pickle
from torch.utils.data import DataLoader
import random
import torchvision.transforms as transforms
from vocab import *
from PIL import Image
import nltk
import numpy as np
from collections import defaultdict
from glove import *


# derived from PA4
def get_recipe_dataloader(dataset, config_data, collate_fn):
    return DataLoader(dataset, batch_size=config_data['dataset']['batch_size'],
                      shuffle=True,
                      #num_workers=config_data['dataset']['num_workers'],
                      collate_fn=collate_fn,
                      #pin_memory=True
                     )

def get_datasets(config_data, ingd_embed = False, mask = False):
    images_root_dir = config_data['dataset']['images_root_dir']
    img_size = config_data['dataset']['img_size']

    train_file_path = config_data['dataset']['train_pickle']
    val_file_path = config_data['dataset']['val_pickle']
    test_file_path = config_data['dataset']['test_pickle']

    vocab_threshold = config_data['dataset']['vocabulary_threshold'] # TODO
    vocab = load_vocab(train_file_path, vocab_threshold)
    
    ing2index = defaultdict(lambda:0, 
                            pickle.load(open(config_data['dataset']['ingredient_to_index'], 'rb'))
                           )# category 0 is unknown

    # detect collate function
    if not ingd_embed:
        fn = collate_fn
    else:
        if mask:
            fn = collate_fn_masked
        else:
            fn = collate_fn_fullembed
    
    train_data_loader = get_recipe_dataloader(
        RecipeDataset(images_root_dir, train_file_path, vocab, img_size, ing2index, ingd_embed = ingd_embed, mask_ingd = mask)
        , config_data, collate_fn = fn)
    test_data_loader = get_recipe_dataloader(
        RecipeDataset(images_root_dir, test_file_path, vocab, img_size, ing2index, ingd_embed = ingd_embed, mask_ingd = mask)
        , config_data, collate_fn = fn)
    val_data_loader = get_recipe_dataloader(
        RecipeDataset(images_root_dir, val_file_path, vocab, img_size, ing2index, ingd_embed = ingd_embed, mask_ingd = mask)
        , config_data, collate_fn = fn)

    return vocab, train_data_loader, val_data_loader, test_data_loader

def find_img_path(imgid, root = '~/val'):
    return os.path.join(root, imgid[0], imgid[1], imgid[2], imgid[3], imgid)

class RecipeDataset(data.Dataset):
    '''for torch.utils.data.DataLoader'''
    
    def __init__(self, root, pickle_path, vocab, img_size, ing2index, transform=None, ingd_embed = False, mask_ingd = False):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformations.
        """
        with open(pickle_path, 'rb') as f:
            dictionary = pickle.load(f)
        self.ing2index = ing2index
        self.ingd_embed = ingd_embed # use ingd_embedding or not
        self.mask_ingd = mask_ingd # use ingd_embedding or not
        self.n_category = max(list(self.ing2index.values()))+1 # 0 to 3143 total of 3144 numbers
        self.root = root
        self.dict = dictionary
        self.ids = list(dictionary.keys())
        self.max_ingd=16
        self.vocab = vocab
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # TODO, might need to change
        ])

        self.resize = transforms.Compose(
            [transforms.Resize(img_size, interpolation=2), transforms.CenterCrop(img_size)])
        
    def sentence_to_tensor(self, caption):
        ''' given sentence, convert to tokens '''
        
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = [self.vocab('<start>')]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        
        return target
        
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        
        vocab = self.vocab
        ann_id = self.ids[index]
        
        title = self.dict[ann_id]['title'].lower()
        
        ing = ' '.join([i['text'] for i in self.dict[ann_id]['ingredients']]).lower()
        
        # list of ingredients
        if not self.ingd_embed:
            # return a list of ingredient index (categories)
            ing_index = [self.ing2index[i] for i in self.dict[ann_id]['ingredient_list']]
            ing_index_tensor = torch.zeros(self.n_category)
            ing_index_tensor[np.array(ing_index)]=1
        else:
            if len(self.dict[ann_id]['ingredient_list'])==0:
                unmasked_embed = torch.zeros((self.max_ingd, 50))
                masked_embed = unmasked_embed[0,:]
            else:
                ing_embed = torch.tensor(np.stack([ingredient2embedding(i) for i in self.dict[ann_id]['ingredient_list']])).float()#n_ingd*50
                # make same size
                if ing_embed.shape[0]>self.max_ingd:
                    ing_embed = ing_embed[:self.max_ingd, :]
                else:
                    padding = torch.zeros((self.max_ingd-ing_embed.shape[0], 50))
                    ing_embed = torch.cat((ing_embed, padding), axis = 0)

                if self.mask_ingd:
                    n_ingd_in_sample = len(self.dict[ann_id]['ingredient_list'])
                    max_size = min([n_ingd_in_sample, self.max_ingd])
                    selected_index = random.choice(list(range(max_size)))

                    masked_embed = ing_embed[selected_index, :]
                    # mask
                    ing_embed[selected_index, :] = 0
                    unmasked_embed = ing_embed
                
                
            
                
            # select 1 ingredient to mask
        
        ins = ' '.join([i['text'] for i in self.dict[ann_id]['instructions']]).lower()
        
        
        img_id = random.choice(self.dict[ann_id]['images'])['id'] # can have multiple images, choose 1
        
        
        path = find_img_path(img_id, root = self.root);
        image = Image.open(path).convert('RGB')
        image = self.resize(image)
        image = self.normalize(np.asarray(image))

        # Convert caption (string) to word ids.
        title_tensor = self.sentence_to_tensor(title)
        ing_tensor = self.sentence_to_tensor(ing)
        ins_tensor = self.sentence_to_tensor(ins)
        
        if self.ingd_embed and self.mask_ingd:
            return image, unmasked_embed, masked_embed
        elif self.ingd_embed and not self.mask_ingd:
            return image, ing_embed # entire ing
        else:
            return image, ing_index_tensor, title_tensor, ing_tensor, ins_tensor, img_id

    def __len__(self):
        return len(self.ids)

def make_same_length(captions):
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return targets
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)
    by padding the captions to make them of equal length.
    We can not use default collate_fn because variable length tensors can't be stacked vertically.
    We need to pad the captions to make them of equal length so that they can be stacked for creating a mini-batch.
    Read this for more information - https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, ing_tensor, title, ing, ins, img_ids = zip(*data)
    
    images = torch.stack(images, 0)
    
    
    titles = make_same_length(title)
    ings = make_same_length(ing)
    inss = make_same_length(ins)
    
    return images, ing_tensors, titles, ings, inss, img_ids
def collate_fn_masked(data):
    """Creates mini-batch tensors from the list of tuples (image, caption) """
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, unmask_embed, masked_embed = zip(*data)
    images = torch.stack(images, 0)
    masked_embed = torch.stack(masked_embed, 0)
    unmasked_embed = torch.stack(unmask_embed, 0)
    
    return images, unmasked_embed, masked_embed

def collate_fn_fullembed(data):
    """Creates mini-batch tensors from the list of tuples (image, caption) """
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, embed = zip(*data)
    images = torch.stack(images, 0)
    embeds = torch.stack(embed, 0)
    
    
    return images, embeds