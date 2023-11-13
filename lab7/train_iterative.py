import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import os
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
from d2l import torch as d2l
import pickle
from dl_assignment_7_common import *
import sys


net = create_lenet()
train_data, valid_data, test_data = get_dataloaders(
    "MNIST", model_name="lenet", batch_size=60)
datasets = {"valid": valid_data, "test": test_data, "train": train_data}
lr, optimizer, momentum = get_lr_optimizer("lenet")
epochs = get_num_epochs("lenet")
early_stop_metric = "valid_loss"
save_patience = 1

if sys.argv[1] == "reinit":
    print("Iterative reinit weights")
    results_model, results_stat = iterative_pruning_training(net, "lenet", datasets, "iterative_reinit", "iterative", True, lr=lr, optimizer=optimizer, momentum=momentum, epochs=epochs, early_stop_metric=early_stop_metric, save_patience=save_patience, plot=False)
    with open("results_stat_iterative_reinit.pkl", "wb") as f:
        pickle.dump(results_stat, f)
else:
    print("Iterative pruning no reset")
    results_model, results_stat = iterative_pruning_training(net, "lenet", datasets, "iterative_reinit", "iterative", False, lr=lr, optimizer=optimizer, momentum=momentum, epochs=epochs, early_stop_metric=early_stop_metric, save_patience=save_patience, plot=False)
    with open("results_stat_iterative.pkl", "wb") as f:
        pickle.dump(results_stat, f)