import numpy as np
import random

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from easyfsl.samplers import TaskSampler
from easyfsl.methods import PrototypicalNetworks

from FSLDataset import FSLDataset
from FSLMethods import training_epoch, evaluate_model
from FSLNetworks import FeatureExtractor
from show_metrics import show_metrics


class FSLTrainer:
    def __init__(self, train_set: FSLDataset, validation_set: FSLDataset, test_set: FSLDataset, config):
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.config = config

    ## Main method for the whole training process, initialized with configured hyperparams
    def train (self, curr_config, metric = 'recall'):

        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        n_way = 2
        n_shot = curr_config['n_shot']
        n_query = 10
        n_tasks_per_epoch = 1
        n_validation_tasks = 100
        
        ## Sampliers used to generate tasks
        train_sampler = TaskSampler(dataset = self.train_set, n_way = n_way, n_shot = n_shot, 
                                    n_query = n_query, n_tasks = n_tasks_per_epoch)
        validation_sampler = TaskSampler(dataset = self.validation_set, n_way = n_way, n_shot = n_shot,
                                        n_query = n_query, n_tasks = n_validation_tasks)
        
        ## Loader generates an iterable given a dataset and a sampler
        train_loader = DataLoader(dataset = self.train_set, batch_sampler = train_sampler, pin_memory = True,
                                collate_fn = train_sampler.episodic_collate_fn)
        validation_loader = DataLoader(dataset = self.validation_set, batch_sampler = validation_sampler, pin_memory = True,
                                    collate_fn = validation_sampler.episodic_collate_fn)
        
        loss_fn = nn.CrossEntropyLoss()

        ## Scheduler: Scales learning rate by gamma at the designated milestones
        scheduler_milestones = [70, 85]
        scheduler_gamma = 0.1

        ## Optimizer
        backbone = FeatureExtractor(in_dim = len(self.train_set.dataframe.columns) - 1, hidden_dim = 256, out_dim = curr_config['embedding_size'])
        model = PrototypicalNetworks(backbone, use_softmax = True)

        learning_rate = 0.001
        momentum = 0.9
        decay = 5e-4
        train_optimizer = optim.SGD(params = model.parameters(), lr = learning_rate, momentum = momentum, 
                                    weight_decay = decay)
        train_scheduler = MultiStepLR(optimizer = train_optimizer, milestones = scheduler_milestones,
                                    gamma = scheduler_gamma)
        
        ## Writer
        log_dir = 'fsl_logs'
        tb_writer = SummaryWriter(log_dir = log_dir)

        ## Train the model
        n_epochs = 10

        ## Track best parameters (weights and biases) and performance of model
        best_state = model.state_dict()
        best_metrics = FSLMetrics()

        print(f'Training {n_shot}-shot classifier with size {curr_config["embedding_size"]} embedding... ...')
        for epoch in range(n_epochs):
            # print(f'Epoch: {epoch}')
            
            average_epoch_loss = training_epoch(model, train_loader, train_optimizer, loss_fn, disable_tqdm = True)

            actuals, predictions, accuracy, precision, recall, specificity, f1_score, auc = evaluate_model(model, validation_loader, disable_tqdm = True)
            
            if metric == 'f1_score':
                if f1_score > best_metrics.f1_score:
                    best_metrics.update(actuals, predictions, accuracy, precision, recall, specificity, f1_score, auc)
                    best_state = model.state_dict()
            elif metric == 'recall':
                if recall > best_metrics.recall:
                    best_metrics.update(actuals, predictions, accuracy, precision, recall, specificity, f1_score, auc)
                    best_state = model.state_dict()
            else:
                raise NotImplementedError

            tb_writer.add_scalar("Train/loss", average_epoch_loss, epoch)
            tb_writer.add_scalar('F1', f1_score, epoch)
            tb_writer.add_scalar('Recall', recall, epoch)

            ## Update the scheduler such that it knows when to adjust the learning rate
            train_scheduler.step()

        ## Retrieve the best model
        _, _ = model.load_state_dict(best_state)
        # print(f'Best f1-score after {n_epochs} epochs of training: {best_f1_score}')
        # print(f'Best recall after {n_epochs} epochs of training: {best_recall}')

        return model, best_metrics


    ## Method to tune hyperparameters
    def tune (self, metric = 'recall', show_metric = False):

        assert 'n_shot' in self.config and 'embedding_size' in self.config
        assert type(self.config['n_shot']) == list and type(self.config['embedding_size']) == list

        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        results = {} ## Key: Value = (k, embedding_size): (metric, model_params)
        best_config = ()
        best_metrics = FSLMetrics()

        ## Manual grid search
        for curr_k in self.config['n_shot']:
            for curr_size in self.config['embedding_size']:
                curr_config = {'n_shot': curr_k, 'embedding_size': curr_size}
                curr_model, curr_metrics = self.train(curr_config, metric)

                if metric == 'recall':
                    if curr_metrics.recall > best_metrics.recall:
                        best_config = (curr_config['n_shot'], curr_config['embedding_size'])
                        best_metrics = curr_metrics
                elif metric == 'f1_score':
                    if curr_metrics.f1_score > best_metrics.f1_score:
                        best_config = (curr_config['n_shot'], curr_config['embedding_size'])
                        best_metrics = curr_metrics
                else:
                    raise NotImplementedError

                results[(curr_k, curr_size)] = (curr_metrics, curr_model.state_dict())
        
        print('########### Tuning complete ###########')
        print(f'Best trial config: k = {best_config[0]}, embedding size = {best_config[1]}')
        print(f'Best trial validation {metric}: {getattr(best_metrics, metric)}')

        if show_metric == True:
            show_metrics(actual = best_metrics.actuals, predicted = best_metrics.predictions, pos_label = 1, neg_label = 0)

        return results, best_config
    

    ## Method to evaluate best model on test set
    def test (self, model_state, config):

        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        
        n_way = 2
        n_shot = config['n_shot']
        n_query = 10
        n_test_tasks = 100
        
        ## Preparing data
        test_sampler = TaskSampler(dataset = self.test_set, n_way = n_way, n_shot = n_shot,
                                    n_query = n_query, n_tasks = n_test_tasks)
        test_loader = DataLoader(dataset = self.test_set, batch_sampler = test_sampler, pin_memory = True,
                                collate_fn = test_sampler.episodic_collate_fn)
        
        ## Preparing model
        backbone = FeatureExtractor(in_dim = len(self.test_set.dataframe.columns) - 1, hidden_dim = 256, out_dim = config['embedding_size'])
        model = PrototypicalNetworks(backbone, use_softmax = True)

        model.load_state_dict(model_state)

        ## Evaluating the model
        test_metrics = FSLMetrics()
        actuals, predictions, accuracy, precision, recall, specificity, f1_score, auc = evaluate_model(model, test_loader)
        test_metrics.update(actuals, predictions, accuracy, precision, recall, specificity, f1_score, auc)

        ## Results
        # print('########### Test results ###########')
        # print(f'F1-score: {f1_score}')
        # print(f'Recall: {recall}')

        return test_metrics
    

class FSLMetrics:
    def __init__(self):
        self.actuals = []
        self.predictions = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.specificity = 0
        self.f1_score = 0
        self.auc = 0

    def update (self, actuals, predictions, accuracy, precision, recall, 
                specificity, f1_score, auc):
        self.actuals = actuals
        self.predictions = predictions
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.specificity = specificity
        self.f1_score = f1_score
        self.auc = auc
