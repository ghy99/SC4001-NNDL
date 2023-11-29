import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import shutil
import time


def retrieve_acc(filename):
    """
    This function retrieves the training and validation accuracies from the training. 
    The values are stored in text files, so we retrieved the values and appended it to lists.

    arguments:
    filename (str): the filepath/filename to the file with the accuracy values.

    returns:
    train_acc (list): list of training accuracies
    val_acc (list): list of validation accuracies
    """
    train_acc = []
    val_acc = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("train Loss"):
                train_acc.append(float(line.split(":")[-1]))
                # print(line)
            if line.startswith("val Loss"):
                val_acc.append(float(line.split(":")[-1]))
            # print(line)

    return train_acc, val_acc

if __name__ == "__main__":
    og_resnet18_train, og_resnet18_val = retrieve_acc("./original_resnet18.txt")
    print(f"og_resnet18_train: {max(og_resnet18_train)}\t\tog_resnet18_val: {max(og_resnet18_val)}")

    og_resnet18_frozen_train, og_resnet18_frozen_val = retrieve_acc("./original_resnet18_frozen.txt")
    print(f"og_resnet18_frozen_train: {max(og_resnet18_frozen_train)}\tog_resnet18_frozen_val: {max(og_resnet18_frozen_val)}")

    deformed_aft_conv_train, deformed_aft_conv_val = retrieve_acc("./deformed_aft_conv1.txt")
    print(f"deformed_aft_conv_train: {max(deformed_aft_conv_train)}\t\tdeformed_aft_conv_val: {max(deformed_aft_conv_val)}")

    deformed_plus_conv_train, deformed_plus_conv_val = retrieve_acc("./deformed_plus_conv1.txt")
    print(f"deformed_plus_conv_train: {max(deformed_plus_conv_train)}\tdeformed_plus_conv_val: {max(deformed_plus_conv_val)}")

