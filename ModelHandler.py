from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from caption_utils import *
from dataloader import get_datasets
from file_utils import *
from model_factory import get_model
from classification_metrics import *
from nlp_metric import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_STATS_DIR = './experiment_data'

def label2onehot(labels, vocab_size):

    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    one_hot = torch.FloatTensor(labels.size(0), labels.size(1), vocab_size+1).zero_().to(device)
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    # remove pad position
    one_hot = one_hot[:, :-1]
    # eos position is always 0
    one_hot[:, 2] = 0

    return one_hot

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class ModelHandler(object):
    def __init__(self, config_name, verbose_freq=2000, weighted = False):
        self.config_data = read_file_in_dir('./config/', config_name + '.json')
        
        self.label_smoothing = 0.1
        self.pad_value = 0

        if self.config_data is None:
            raise Exception("Configuration file doesn't exist: ", config_name)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.__name = self.config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        self.verbose_freq = verbose_freq

        # Load Datasets
        self.__vocab, self.indg_vocab, self.__train_loader, self.__val_loader, self.__test_loader, self.train_dataset, self.test_dataset, self.val_dataset = get_datasets(self.config_data)

        # Setup Experiment
        self.__generation_config = self.config_data['generation']
        self.__epochs = self.config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []

        # Save your best model in this field and use this in test method.
        self.__best_model = None

        # Init Model
        self.__model = get_model(self.config_data, self.__vocab, self.indg_vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.weighted = weighted
        if weighted:
            self.calculate_weight()
            self.__criterion = self.__model.get_loss_criteria(**{'reduction':'none'})
        else:
            self.__criterion = self.__model.get_loss_criteria()
        self.crit_eos = self.__model.get_loss_criteria()
        self.__optimizer = torch.optim.Adam(
            self.__model.parameters(), lr=self.config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()
        
        

    # Loads the experiment data if exists to resume training from last saved checkpoint.

    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(
                self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(
                self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(
                self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.to(self.device)
            # self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        # early stopping
        try:
            min_val_loss = min(self.__val_losses)
        except:
            min_val_loss = 100

        start_epoch = self.__current_epoch
        # loop over the dataset multiple times
        for epoch in range(start_epoch, self.__epochs):
            start_time = datetime.now()
            self.__current_epoch = epoch
            if self.__current_epoch > self.config_data["n_fine_tune"]:
                fine_tune = False
                print('start trainingin entire network')
            else:
                fine_tune = True
            train_loss = self.__train(fine_tune= fine_tune)
            val_loss = self.__val()

            if val_loss < min_val_loss:
                # save best model
                self.__save_best_model()
                min_val_loss = val_loss

            print(f'epoch {epoch}, train:{train_loss}, val:{val_loss}')
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def calculate_weight(self, pseudocount = 1.1):
        if not os.path.isfile('weights.pt'):
            for i, (images, title, ing_binary, ing, ins, ann_id) in enumerate(self.__train_loader):
                if i == 0:
                    count = ing_binary.sum(axis = 0)
                else:
                    count += ing_binary.sum(axis = 0)

                weight = 1/np.log(count+pseudocount)
                weight = weight/weight.sum()

                self.class_weight = weight
            torch.save(weight, 'weights.pt')
        else:
            self.class_weight = torch.load('weights.pt')
            print('load precalc category weights')
        self.class_weight = self.class_weight.to(self.device)
            
        
    def __train(self, fine_tune=False):
        print(f'training on {self.device}')
        self.__model.train()
        training_loss = 0
        train_loss_epoch = []
        for i, (images, title, ing_binary, ing, ins, ann_id) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            
            input_dict, output_dict = self.get_input_and_target(images, title, ing_binary, ing, ins)
            target_ingrs = output_dict[self.__model.input_outputs['output'][0]] # only 1 output            
            ingr_probs, ing_idx, eos = self.__model(input_dict, fine_tune=fine_tune)


            # target_ingrs = target['ingredient']
            target_one_hot_smooth = label2onehot(target_ingrs, len(self.indg_vocab))
            target_one_hot_smooth[target_one_hot_smooth == 1] = (1-self.label_smoothing)
            target_one_hot_smooth[target_one_hot_smooth == 0] = self.label_smoothing / target_one_hot_smooth.size(-1)

            ingr_loss = self.__criterion(ingr_probs, target_one_hot_smooth)
            ingr_loss = torch.mean(ingr_loss, dim=-1)

            train_loss_epoch.append(ingr_loss.item())


            target_eos = ((target_ingrs == 2) ^ (target_ingrs == self.pad_value))
            eos_pos = (target_ingrs == 0)
            eos_head = ((target_ingrs != self.pad_value) & (target_ingrs != 0))
            eos_loss = self.crit_eos(eos, target_eos.float())

            mult = 1/2
            # eos loss is only computed for timesteps <= t_eos and equally penalizes 0s and 1s
            eos_loss = mult*(eos_loss * eos_pos.float()).sum(1) / (eos_pos.float().sum(1) + 1e-6) + \
                                 mult*(eos_loss * eos_head.float()).sum(1) / (eos_head.float().sum(1) + 1e-6)
            

            ingr_loss = ingr_loss.mean()
            eos_loss = eos_loss.mean()
            training_loss = 0.8 * ingr_loss + 0.2*eos_loss
            # training_loss = self.__criterion(ing_prob, target)
            # if self.weighted:
            #     training_loss = (training_loss*self.class_weight).mean()
                
            # train_loss_epoch.append(training_loss.item())

            if i % self.verbose_freq == 0:
                print(f'batch{i}, {training_loss}')
            training_loss.backward()
            self.__optimizer.step()

        return np.mean(train_loss_epoch)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0
        val_loss_epoch = []

        with torch.no_grad():
            for i, (images, title, ing_binary, ing, ins, img_id) in enumerate(self.__val_loader):
                input_dict, output_dict = self.get_input_and_target(images, title, ing_binary, ing, ins)
                target = output_dict[self.__model.input_outputs['output'][0]] # only 1 output            
                pred = self.__model(input_dict)


                val_loss = self.__criterion(pred, target)
                if self.weighted:
                    val_loss = (val_loss*self.class_weight).mean()
                val_loss_epoch.append(val_loss.item())

        return np.mean(val_loss_epoch)

    
    def get_raw_data(self, ann_ids, pred):
        ''' given a list of ann_ids, pred_word_index, return the raw data and do basic NLP preprocessing like lower()
        ann_ids: a tuple containing data ID
        pred: [batch_size * max_len] containing ingredient idx
        ''' 
        
        data = []
        # make to words
        pred_word = np.vectorize(lambda i: self.indg_vocab.idx2word[i])(pred)
        
        # get raw data
        for i, ann_id in enumerate(ann_ids):
            title,  ingridients, instructions,img_paths = self.test_dataset.get_raw_data(ann_id)
            p = pred_word[i]

            # find <start> and <end> token and extract in between
            try:
                end = p.index('<end>')

            except:
                end = len(p)
            try:
                start = p.index('<start>')
            except:
                start = 0
                
            # sort the ingredients to achieve highest possible BLEU score
            
            
            ingridients.sort()


            data.append([title, instructions, ingridients, img_paths, p])
        # make prediction into words

        data = pd.DataFrame(data, columns = ['title', 'instructions', 'ingredients', 'img_paths', 'predicted_ingredients'])
        
        data['predicted_ingredients_unique'] = data['predicted_ingredients'].apply(
            lambda x: sorted(list(set(x))))
        data['ing_sentence'] = data['ingredients'].apply(lambda x: (' '.join(x)).lower())        
        data['pred_ing_sentence'] = data['predicted_ingredients_unique'].apply(
            lambda x: (' '.join(x)).lower())
        
        # calculate BLEU scores
        data['bleu1'] = data.apply(lambda x: bleu1(x['ing_sentence'], x['pred_ing_sentence']), axis = 1)
        data['bleu4'] = data.apply(lambda x: bleu4(x['ing_sentence'], x['pred_ing_sentence']), axis = 1)
        
        # jaccard index
        data['jaccard'] = data.apply(lambda x: jaccard(x['ingredients'], x['predicted_ingredients']), axis = 1)
                                     
        
        return data
        
        
    def return_example(self, use_best_model=True, mode='stochastic', gamma=0.1):
        ''' you can specify mode and visualize  '''
        self.__model.eval()

        if use_best_model:  # use those from early stop
            print('=== Using best model from early stop ===')
            best_model_path = os.path.join(
                self.__experiment_dir, 'best_model.pt')
            self.__model.load_state_dict(torch.load(best_model_path))
            
        
        with torch.no_grad():
            
            for iter_, (images, title, ing_binary, ing, ins, ann_ids) in enumerate(self.__test_loader):
                
                input_dict, output_dict = self.get_input_and_target(images, title, ing_binary, ing, ins)
                target = output_dict[self.__model.input_outputs['output'][0]] # only 1 output            
                _, pred = self.__model.predict(input_dict, mode = mode, r = gamma) 
                pred = pred.detach().cpu().numpy()
                
                
                break
            
            data = self.get_raw_data(ann_ids, pred)
                
            
            # use ann_id to extract images
        return data

    # TODO: Not yet implemented properly
    def get_best_model(self):
        
        best_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
        self.__model.load_state_dict(torch.load(best_model_path))
        
        return self.__model
    
    def index2binary(self, word_index, n_ingredients):
        ''' convert [2,5,9,0] to binary matrix, for classification metrics calculation 
        word_index = [n_batch, max_len]
        return [n_batch, n_ingredients]
        '''
        batch_size, max_len = word_index.shape
        binary_matrix = np.zeros((batch_size, n_ingredients))
        j = word_index.flatten().cpu()
        i = np.stack([np.arange(batch_size)]*max_len, axis = 1).flatten()
        
        binary_matrix[i,j] = 1
        
        return binary_matrix
        
    def test(self, use_best_model=True):
        ''' return classification metric, BLEU score, and other quanlitative metrics '''
        self.__model.eval()
        
        if use_best_model: # use those from early stop
            print('=== Using best model from early stop ===')
            
            self.get_best_model()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0
        test_loss_epoch = []
        scores = []
        losses = {}
        with torch.no_grad():
            b1s = []
            b4s = []
            js = []
            for iter_, (images, title, ing_binary, ing, ins, ann_ids) in enumerate(self.__test_loader):
                
                input_dict, output_dict = self.get_input_and_target(images, title, ing_binary, ing, ins)
                target = output_dict[self.__model.input_outputs['output'][0]] # only 1 output            
                ingr_probs, pred_word_index, eos = self.__model.predict(input_dict)
                

                
                losses['ingr_loss'] = ingr_loss

                # cardinality penalty
                losses['card_penalty'] = torch.abs((ingr_probs*target_one_hot).sum(1) - target_one_hot.sum(1)) + \
                                        torch.abs((ingr_probs*(1-target_one_hot)).sum(1))

                eos_loss = self.crit_eos(eos, target_eos.float())

                mult = 1/2
                # eos loss is only computed for timesteps <= t_eos and equally penalizes 0s and 1s
                losses['eos_loss'] = mult*(eos_loss * eos_pos.float()).sum(1) / (eos_pos.float().sum(1) + 1e-6) + \
                                    mult*(eos_loss * eos_head.float()).sum(1) / (eos_head.float().sum(1) + 1e-6)
                # iou
                pred_one_hot = label2onehot(ingr_ids, self.pad_value)
                # iou sample during training is computed using the true eos position
                losses['iou'] = softIoU(pred_one_hot, target_one_hot)










                # Calculate loss
                # test_loss = self.__criterion(pred, target)
                # if self.weighted:
                #     test_loss = (test_loss*self.class_weight).mean()
                
                # test_loss_epoch.append(test_loss.item())
                
                
                # F1 score and such
                binary_matrix = self.index2binary(pred_word_index, ing_binary.shape[1])
                evl=calculate_metrics(binary_matrix, ing_binary.cpu().detach().numpy(), self.indg_vocab)
                evl = evl.replace(0, np.nan) # when no class is there
                scores.append(evl.values)

                
                # BLEU score
                raw_data = self.get_raw_data(ann_ids, pred_word_index.detach().cpu().numpy())
                b1s.append(raw_data['bleu1'].mean())
                b4s.append(raw_data['bleu4'].mean())
                js.append(raw_data['jaccard'].mean())
                
                
                   
        
        mean_test_loss = np.mean(test_loss_epoch)
        mean_evl = np.nanmean(np.stack(scores), axis = 0)
        evl_df = pd.DataFrame(mean_evl, index = evl.index, columns = evl.columns)
        evl_df.to_csv(os.path.join(self.__experiment_dir, f'cls_eval_{self.__current_epoch}.csv'))
        
        result_str = "Test Performance: Loss: {}".format(mean_test_loss)
        self.__log(result_str)
        stat_df = pd.DataFrame([[mean_test_loss, np.mean(b1s), np.mean(b4s), np.mean(js)]], 
                               columns = ['loss', 'bleu1', 'bleu4', 'jaccard'])
        stat_df.to_csv(os.path.join(self.__experiment_dir, f'scores_{self.__current_epoch}.csv'))
        return mean_test_loss, mean_evl, np.mean(b1s), np.mean(b4s), np.mean(js)


    def __save_model(self):
        if not os.path.exists(self.__experiment_dir):
            os.makedirs(self.__experiment_dir)

        root_model_path = os.path.join(
            self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict,
                      'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __save_best_model(self): 
        if not os.path.exists(self.__experiment_dir):
            os.makedirs(self.__experiment_dir)

        best_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
        torch.save(self.__model.state_dict(), best_model_path)


    def __record_stats(self, train_loss, val_loss):
        if not os.path.exists(self.__experiment_dir):
            os.makedirs(self.__experiment_dir)

        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir,
                             'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir,
                             'val_losses.txt', self.__val_losses)

        write_to_file(os.path.join(self.__experiment_dir,
                                   'config.json'), self.config_data)

    def __log(self, log_str, file_name=None):
        print(log_str)
        if not os.path.exists(self.__experiment_dir):
            os.makedirs(self.__experiment_dir)

        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * \
                             (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()

    def random_mask_ingredient(ing):
        # find all non-zero entries
        # ing: batch_size, arbitrary length
        
        # find positions with ingredients, collate_fn pad 0s
        non_zeros = ing.reshape(-1)[ing.reshape(-1)!=0]
        frequency = Counter(non_zeros.numpy())

        # sampling weights is reciporcal to log frequency in this batch
        index = list(frequency.keys())
        p = 1/np.log((np.array(list(frequency.values()))+1).astype(float))

        sampling_frequency = defaultdict(lambda: 0, 
                                        {index[i]: p[i] for i in range(len(p))})

        # for every example
        to_mask_idx = []
        for row_index in range(ing.shape[0]):

            ingd_one_ex = ing[row_index]
            pos_prob = np.array([sampling_frequency[i] for i in ingd_one_ex.cpu().numpy()])
            # normalize
            pos_prob = pos_prob/pos_prob.sum()

            # sample 1 position based on probability
            to_mask = np.random.choice(np.arange(ing.shape[1]), p = pos_prob)

            to_mask_idx.append(to_mask)

        masked = torch.clone(ing[np.arange(ing.shape[0]), to_mask_idx])
        ing[np.arange(ing.shape[0]), to_mask_idx] = 0 # should we set another token?
        
        
        return masked, ing
    
    
    def get_input_and_target(self, img, title, ing_binary, ing, ins,):
        to_return = {'input':{}, 'output':{}}
        input_target_dictionary= self.__model.get_input_and_target_feature()
        
        
        if 'masked_ingredient' in input_target_dictionary['output']:
            
            masked_ingredient, unmasked = self.random_mask_ingredient(ing) # 2 things
            to_return['input']['unmask_ingredient']=unmasked.to(self.device)
            to_return['output']['masked_ingredient']=masked_ingredient.to(self.device)
        for s in ['input', 'output']:
            
            for item, name in zip([img, title, ing_binary, ing, ins],
                           ['image', 'title', 'ingredient_binary', 'ingredient', 'instruction']):
                if name in input_target_dictionary[s]:
                    
                    to_return[s][name]=item.to(self.device)

        return to_return['input'], to_return['output']