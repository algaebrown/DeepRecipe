################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################
from datetime import datetime
from tabnanny import verbose
from classification_metrics import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from res_attn import *



from dataloader import get_datasets
from file_utils import *
from model_factory import get_model
ROOT_STATS_DIR = './experiment_data'

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name, model, verbose_freq = 2000):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        self.config_data = config_data

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)
        self.verbose_freq = verbose_freq
        # Load Datasets
        self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(config_data, ingd_embed = True, mask = True)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = model

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = torch.nn.MSELoss() # binary cross entropy
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            # self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        # early stopping
        try:
            min_val_loss = min(self.__val_losses)
        except:
            min_val_loss = 100
        
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            if self.__current_epoch > self.config_data["n_fine_tune"]:
                fine_tune = False
                print('start trainingin entire network')
            else:
                fine_tune = True
            train_loss = self.__train(fine_tune = fine_tune)
            val_loss = self.__val()
            
            if val_loss < min_val_loss:
                # save best model
                best_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
                torch.save(self.__model.state_dict(), best_model_path)
                min_val_loss = val_loss

            print(f'epoch {epoch}, train:{train_loss}, val:{val_loss}')
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self, fine_tune = True):
        
        print(f'training on {self.device}')
        self.__model.train()
        training_loss = 0
        train_loss_epoch = []
        
        for i, (images, unmasked_embed, masked_embed) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            
            
            images = images.to(self.device)
            unmasked_embed = unmasked_embed.to(self.device)
            masked_embed = masked_embed.to(self.device)
            
            pred = self.__model(images, unmasked_embed)
                
            training_loss = self.__criterion(pred, masked_embed)
            train_loss_epoch.append(training_loss.item())

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
            for i, (images, unmasked_embed, masked_embed) in enumerate(self.__val_loader):
                images = images.to(self.device)
                unmasked_embed = unmasked_embed.to(self.device)
                masked_embed = masked_embed.to(self.device)
            
                pred = self.__model(images, unmasked_embed)
                
                val_loss = self.__criterion(pred, masked_embed)
                
                
                val_loss_epoch.append(val_loss.item())

        return np.mean(val_loss_epoch)

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    
            
    def test(self, use_best_model = True):
        self.__model.eval()
        
        if use_best_model: # use those from early stop
            print('=== Using best model from early stop ===')
            best_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
            self.__model.load_state_dict(torch.load(best_model_path))
        
        test_loss = 0
        bleu1 = 0
        bleu4 = 0
        test_loss_epoch = []
        scores = []
        with torch.no_grad():
            b1s = []
            b4s = []
            for iter_, (images, unmasked_embed, masked_embed) in enumerate(self.__test_loader):
                images = images.to(self.device)
                unmasked_embed = unmasked_embed.to(self.device)
                masked_embed = masked_embed.to(self.device)
            
                pred = self.__model(images, unmasked_embed)
                
                test_loss = self.__criterion(pred, masked_embed)
                test_loss_epoch.append(test_loss.item())
                
                

                if iter_ % self.verbose_freq == 0:
                    print(f'batch{iter_}, {test_loss}')
                

                
                # TODO:  class specific precision and recall
                
        mean_test_loss = np.mean(test_loss_epoch)
#         mean_evl = np.nanmean(np.stack(scores), axis = 0)
#         evl_df = pd.DataFrame(mean_evl, index = evl.index, columns = evl.columns)
#         evl_df.to_csv(os.path.join(self.__experiment_dir, f'cls_eval_{self.__current_epoch}.csv'))
        
        result_str = "Test Performance: Loss: {}".format(mean_test_loss)
        self.__log(result_str)

        return mean_test_loss

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
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
    def get_loss(self):
        return self.__training_losses, self.__val_losses