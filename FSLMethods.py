import io
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from tqdm import tqdm

from statistics import mean

from show_metrics import show_metrics, get_metrics
from get_processed_data import get_processed_data
from sampling import undersample, oversample, smote
from feature_selection import rf_select
from FSLDataset import FSLDataset


## Method to sample data
def get_sampled_data (X, y, sampling_method):
    new_X, new_y = None, None
    
    if sampling_method == 'oversampling':
        new_X, new_y = oversample(X, y)
    elif sampling_method == 'undersampling':
        new_X, new_y = undersample(X, y)
    else:
        new_X, new_y = smote(X, y)

    return new_X, new_y


## Form the required datasets
def form_datasets (X_train, y_train, X_val, y_val, X_test, y_test, feature_selection = False, sampling_method = ''):
   
    if feature_selection == True:
        ## Select features (indices) using random forest classifier
        features_idx = rf_select(X_train, y_train)
        X_train = X_train[features_idx]
        X_val = X_val[features_idx]
        X_test = X_test[features_idx]

    ## Sampling
    if sampling_method != '':
        X_train, y_train = get_sampled_data(X_train, y_train, sampling_method)

    train_df = X_train
    y_train = y_train
    train_df['Fraud'] = y_train

    validation_df = X_val
    y_val = y_val
    validation_df['Fraud'] = y_val

    test_df = X_test
    y_test = y_test
    test_df['Fraud'] = y_test

    train_set = FSLDataset(train_df)
    validation_set = FSLDataset(validation_df)
    test_set = FSLDataset(test_df)

    return train_set, validation_set, test_set


## Represents one epoch / episode of multiple tasks
    ## Each epoch produces a new model
def training_epoch (model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn, disable_tqdm = False, implementation = 'easyfsl'):
    all_loss = []
    model.train()

    with tqdm(enumerate(data_loader), total = len(data_loader), disable = disable_tqdm) as tqdm_train:
        ## For each task, make prediction, calculate loss, update model params
        for train_task, (support_data, support_labels, query_data, query_labels, _) in tqdm_train:
                       
            # print(f'Starting task {task_idx}')
            # print('support_data: ', support_data)
            # print('support_labels: ', support_labels)
            
            optimizer.zero_grad()

            if implementation == 'easyfsl':
                model.process_support_set(support_data, support_labels)
                classification_scores = model(query_data)
            else:
                classification_scores = model(support_data, support_labels, query_data)
            

            task_loss = loss_fn(classification_scores, query_labels)
            task_loss.backward()
            optimizer.step()

            all_loss.append(task_loss.item())

            tqdm_train.set_postfix(loss = mean(all_loss))
    
    return mean(all_loss)


## Method to evaluate model performance on few shot classification tasks
def evaluate_model (model: nn.Module, data_loader: DataLoader, disable_tqdm = False, device = 'cpu'):
    actual_lst = []
    predicted_lst = []

    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(data_loader), total = len(data_loader), disable = disable_tqdm) as tqdm_eval:
            ## Loop through tasks
            for eval_task, (support_data, support_labels, query_data, query_labels, _) in tqdm_eval:
                model.process_support_set(support_data, support_labels)
                predictions = model.forward(query_data).detach()
                
                ## Append predictions to predicted list
                for pred in predictions:
                    ## Get predicted class
                    pred_class = torch.max(pred, 0)[1].item()
                    predicted_lst.append(pred_class)
                
                ## Append actual query labels to actual list
                for act in query_labels:
                    actual_lst.append(act.item())

                ## Use results to get performance metrics (on rolling basis)
                accuracy, precision, recall, specificity, f1_score = get_metrics(actual_lst, predicted_lst, pos_label = 1, neg_label = 0)

                tqdm_eval.set_postfix(f1 = f1_score, recall = recall)

    return actual_lst, predicted_lst, accuracy, precision, recall, specificity, f1_score


## Method to show performance of best model
def FSL_main_method ():

    return