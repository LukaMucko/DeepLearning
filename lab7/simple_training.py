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
epochs, lr = 2, 0.01

train_data, valid_data, test_data = get_dataloaders(dataset_name)
datasets = {"valid": valid_data, "test": test_data, "train": train_data}

base_net = create_network(net_name, image_size=get_image_size(dataset_name))
trained_net1, history1 = train(base_net, datasets, "simple_training_1", epochs=epochs, lr=lr, plot=False)
assert not (
        base_net.state_dict().__str__() == trained_net1.state_dict().__str__())  # check if the weights are the same
trained_net2, history2 = train(base_net, datasets, "simple_training_2", epochs=epochs, lr=lr, plot=False)

