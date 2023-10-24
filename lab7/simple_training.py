import torch
import torch.nn
import torch.nn.utils.prune
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from d2l import torch as d2l
from dl_assignment_7_common import *

dataset_name = "MNIST"
net_name = "lenet"

train_data, valid_data, test_data = get_dataloaders(dataset_name)
datasets = {"valid": valid_data, "test": test_data, "train": train_data}

net = create_network(net_name, image_size=get_image_size(dataset_name))

train(net, datasets, "simple_training", epochs=10, lr=0.1, plot=False)

#%%

#%%
