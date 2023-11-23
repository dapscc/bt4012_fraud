import os
import numpy as np

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from easyfsl.samplers import TaskSampler
from easyfsl.methods import PrototypicalNetworks

from FSLMethods import training_epoch, evaluate_model, get_datasets_for_tuner
from FSLNetworks import DummyNetwork

import ray
from ray import train, tune
# from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

## Main method for the whole training process, initialized with configured hyperparams
def fsl_trainer (config):

    n_way = 2
    n_shot = config['n_shot']
    n_query = 5
    n_tasks_per_epoch = 1
    n_validation_tasks = 100

    ## TODO: Find a better way to get data...
    train_set, validation_set = get_datasets_for_tuner()
    
    ## Sampliers used to generate tasks
    train_sampler = TaskSampler(dataset = train_set, n_way = n_way, n_shot = n_shot, 
                                n_query = n_query, n_tasks = n_tasks_per_epoch)
    validation_sampler = TaskSampler(dataset = validation_set, n_way = n_way, n_shot = n_shot,
                                    n_query = n_query, n_tasks = n_validation_tasks)
    
    ## Loader generates an iterable given a dataset and a sampler
    train_loader = DataLoader(dataset = train_set, batch_sampler = train_sampler, pin_memory = True,
                            collate_fn = train_sampler.episodic_collate_fn)
    validation_loader = DataLoader(dataset = validation_set, batch_sampler = validation_sampler, pin_memory = True,
                                collate_fn = validation_sampler.episodic_collate_fn)
    
    loss_fn = nn.CrossEntropyLoss()

    ## Scheduler: Scales learning rate by gamma at the designated milestones
    scheduler_milestones = [70, 85]
    scheduler_gamma = 0.1

    ## Optimizer
    backbone = DummyNetwork(in_dim = len(train_set.dataframe.columns) - 1, hidden_dim = 256, out_dim = config['embedding_size'])
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
    n_epochs = 100
    actuals = []
    predictions = []

    ## Track best parameters (weights and biases) and performance of model
    best_state = model.state_dict()
    best_f1_score = 0.0
    best_recall = 0.0

    print(f'Training {n_shot}-shot classifier with size {config["embedding_size"]} embedding... ...')
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch}')
        
        average_epoch_loss = training_epoch(model, train_loader, train_optimizer, loss_fn)

        actuals, predictions, _, _, recall, _, f1_score = evaluate_model(model, validation_loader)
        
        # if f1_score > best_f1_score:
        #     best_f1_score = f1_score
        #     best_state = model.state_dict()
        #     print("Ding ding ding! We found a new best model!")

        if recall > best_recall:
            best_recall = recall
            best_f1_score = f1_score
            best_state = model.state_dict()
            print("Ding ding ding! We found a new best model!")

        tb_writer.add_scalar("Train/loss", average_epoch_loss, epoch)
        tb_writer.add_scalar('F1', f1_score, epoch)
        tb_writer.add_scalar('Recall', recall, epoch)

        ## Update the scheduler such that it knows when to adjust the learning rate
        train_scheduler.step()

        ## Save a Ray Tune checkpoint
        os.makedirs('fsl_model', exist_ok = True)
        torch.save((best_state, train_optimizer.state_dict()), 'fsl_model/checkpoint.pt')
        # checkpoint = train.Checkpoint.from_directory('fsl_model')
        train.report({
            'f1': best_f1_score,
            'recall': best_recall
        })

    ## Retrieve the best model
    _, _ = model.load_state_dict(best_state)
    print(f'Best f1-score after {n_epochs} epochs of training: {best_f1_score}')
    print(f'Best recall after {n_epochs} epochs of training: {best_recall}')


    return actuals, predictions


## Method to tune hyperparameters
def fsl_tuner (trainer, config, metric = 'recall', num_samples = 1):

    ## Cuts off a trial if performance is poor (metric to be specified)
    # scheduler = ASHAScheduler()

    tuner = tune.Tuner(
        trainable = tune.with_resources(
            tune.with_parameters(trainer),
            resources = {'cpu': 2}
        ),
        tune_config = tune.TuneConfig(
            metric = metric,
            mode = 'max',
            # scheduler = scheduler,
            num_samples = num_samples
        ),
        param_space = config
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric, 'max')

    print('Best trial config', best_result.config)
    print('Best trial validation ' + metric + ': ', best_result.metrics[metric])

    ## TODO: Test best model

    return
