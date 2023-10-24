# This is a file where you should put your own functions

import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

# TODO: Datasets go here.

def get_dataloaders(dataset, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
    ])

    if dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
        test_data = torchvision.datasets.MNIST("data/mnist", train=False, download=True, transform=transform)

    elif dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10("data/cifar10", train=True, download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10("data/cifar10", train=False, download=True, transform=transform)

    elif dataset == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST("data/fashionmnist", train=True, download=True,
                                                          transform=transform)
        test_data = torchvision.datasets.FashionMNIST("data/fashionmnist", train=False, download=True,
                                                      transform=transform)
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    valid_size = 5000
    train_size = len(train_dataset) - valid_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
    test = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
    return train, valid, test


# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------

# TODO: Define network architectures here

def create_lenet(image_size=28):
    return torch.nn.Sequential(
        torch.nn.Linear(image_size ** 2, 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    )


def create_conv_2(image_size=32):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Flatten(),
        torch.nn.Linear(int((image_size / 2) ** 2 * 64), 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )


def create_conv_4(image_size=32):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Flatten(),
        torch.nn.Linear(int((image_size / 4) ** 2 * 128), 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )


def create_conv_6(image_size=32):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Flatten(),
        torch.nn.LazyLinear(256),
        torch.nn.ReLU(),
        # torch.nn.Linear(int((image_size/8)**2 * 256), 256),
        # torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding='same'):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, X):
        Y = torch.nn.functional.relu(self.conv1(X))
        Y = self.conv2(Y)
        Y += X
        return torch.nn.functional.relu(Y)


def create_resnet_18(image_size=32):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),

        ResidualBlock(16, 16),
        ResidualBlock(16, 16),
        ResidualBlock(16, 16),

        ResidualBlock(16, 32, strides=2),
        ResidualBlock(32, 32),
        ResidualBlock(32, 32),

        ResidualBlock(32, 64, strides=2),
        ResidualBlock(64, 64),
        ResidualBlock(64, 64),

        torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Linear(int((image_size / 8) ** 2 * 64), 10),
    )


def create_vgg_19(image_size=32):
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),

        torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        torch.nn.Linear(int((image_size / 32) ** 2 * 64), 10),
    )


def create_network(arch, **kwargs):
    # TODO: Change this function for the architectures you want to support
    if arch == 'lenet':
        return create_lenet(**kwargs)
    elif arch == 'conv2':
        return create_conv_2(**kwargs)
    elif arch == 'conv4':
        return create_conv_4(**kwargs)
    elif arch == 'conv6':
        return create_conv_6(**kwargs)
    elif arch == 'resnet18':
        return create_resnet_18(**kwargs)
    elif arch == 'vgg19':
        return create_vgg_19(**kwargs)
    else:
        raise Exception(f"Unknown architecture: {arch}")


# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

# TODO: Define training, testing and model loading here

# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

# TODO: Put functions related to pruning here
def local_prune(net, fc_amount, conv_amount, out_amount, prune_operation):
    for layer in list(net.children())[:-1]:
        if isinstance(layer, torch.nn.Linear):
            prune_operation(layer, name="weight", amount=fc_amount)
        elif isinstance(layer, torch.nn.Conv2d):
            prune_operation(layer, name="weight", amount=conv_amount)
    prune_operation(list(net.children())[-1], name="weight", amount=out_amount)


def global_prune(net, amount, prune_type, layer_type_to_prune=torch.nn.Conv2d):
    parameters = []
    for layer in net.children():
        if isinstance(layer, layer_type_to_prune):
            parameters.append((layer, "weight"))
        elif isinstance(layer, ResidualBlock):
            for sub_layer_name, sub_layer in layer.named_children():
                if isinstance(sub_layer, layer_type_to_prune):
                    parameters.append((sub_layer, "weight"))

    prune.global_unstructured(parameters, prune_type, amount=amount)


def prune_network(net, net_type, amount, prune_type=prune.L1Unstructured, conv_amount=None, out_amount=None):
    local = net_type in ["lenet", "conv2", "conv4", "conv6"]
    if local:
        if conv_amount is None:
            conv_amount = amount
        if out_amount is None:
            out_amount = amount / 2
        local_prune(net, amount, conv_amount, out_amount, prune_type.apply)
    else:
        global_prune(net, amount, prune_type)


def prune_network_from_mask(net_in, net_out):
    for layer_in, layer_out in zip(net_in.children(), net_out.children()):
        if isinstance(layer_in, (torch.nn.Linear, torch.nn.Conv2d)):
            prune.custom_from_mask(layer_out, name="weight", mask=layer_in.weight_mask)
        elif isinstance(layer_in, ResidualBlock):
            for sublayer_in, sublayer_out in zip(layer_in.children(), layer_out.children()):
                if isinstance(sublayer_in, (torch.nn.Linear, torch.nn.Conv2d)):
                    prune.custom_from_mask(sublayer_out, name="weight", mask=sublayer_in.weight_mask)

# %%
