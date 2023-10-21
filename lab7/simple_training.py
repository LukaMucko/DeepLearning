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

train, valid, test = get_dataloaders("MNIST")
image_size = len(train.dataset.__getitem__(0)[0])**(1/2)

net = create_network("lenet", image_size=image_size)
#%%
