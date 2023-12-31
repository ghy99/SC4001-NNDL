### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



def split_dataset(df, columns_to_drop, test_size, random_state):
    label_encoder = preprocessing.LabelEncoder()

    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

'''
Added code from Part A1
copied MLP, Custom Dataset
'''
from torch.nn.modules import dropout
from torch.nn.modules.activation import Sigmoid

class MLP(nn.Module):
    def __init__(self, no_features, no_hidden, no_labels):
        """
        Linear neuron from input to first hidden layer with
        ReLU activation function and Dropout of 0.2

        Linear neuron from first hidden layer to second hidden layer with
        ReLU activation function and Dropout of 0.2

        Linear neuron from 2nd hidden layer to 3rd hidden layer with
        ReLU activation function and Dropout of 0.2

        Linear neuron from 3rd hidden layer to output layer with
        Sigmoid activation function.
        """
        super().__init__()
        # Uncomment this for part A3
        # next_hidden = 128
        next_hidden = 256
        self.mlp_stack = nn.Sequential(
            # YOUR CODE HERE
            nn.Linear(no_features, no_hidden),
            # nn.Linear(no_features, first_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),

            # nn.Linear(no_hidden, no_hidden),
            nn.Linear(no_hidden, next_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),

            # nn.Linear(no_hidden, no_hidden),
            nn.Linear(next_hidden, next_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # nn.Linear(no_hidden, no_labels),
            nn.Linear(next_hidden, no_labels),
            nn.Sigmoid(),
        )
    # YOUR CODE HERE
    def forward(self, x):
      """
      Added a forward(x) function to return logits.
      """
      logits = self.mlp_stack(x)
      return logits


class CustomDataset(Dataset):
    # YOUR CODE HERE
    """
    Implemented the necessary functions inherited from Dataset class.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
      return self.X[idx], self.y[idx]

loss_fn = nn.CrossEntropyLoss()