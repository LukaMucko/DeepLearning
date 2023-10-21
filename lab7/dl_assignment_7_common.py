# This is a file where you should put your own functions

import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms



# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

# TODO: Datasets go here.

def get_dataloaders(dataset, model_name="lenet", batch_size=64):
    if not model_name.startswith("lenet"):
        transform = transforms.Compose([transforms.ToTensor()])
    else:
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
        train_dataset = torchvision.datasets.FashionMNIST("data/fashionmnist", train=True, download=True, transform=transform)
        test_data = torchvision.datasets.FashionMNIST("data/fashionmnist", train=False, download=True, transform=transform)
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
        torch.nn.Linear(int(image_size**2), 300),
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
        torch.nn.Linear(int((image_size/2)**2 * 64), 256),
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
        torch.nn.Linear(int((image_size/4)**2 * 128), 256),
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
        torch.nn.Conv2d(3, 16, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),

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
        torch.nn.Linear(int((image_size/8)**2 * 64), 10),
    )

def create_vgg_19(image_size=32):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=(3, 3), padding='same'), torch.nn.ReLU(),
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
        torch.nn.Linear(int((image_size/32)**2 * 64), 10),
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

def record_metrics(model, epoch_stats, datasets, loss_fn):
    with torch.no_grad():
        for name, dataset in datasets.items():
            eval_metric = d2l.Accumulator(2)
            for x, y in dataset:
                x, y = x.to(device), y.to(device, torch.long)
                y_hat = model(x)
                loss = loss_fn(y_hat, y).item()
                eval_metric.add(loss * x.shape[0], x.shape[0])
            epoch_stats[name + "_loss"].append(eval_metric[0] / eval_metric[1])
            epoch_stats[name + "_acc"].append(d2l.evaluate_accuracy_gpu(model, dataset))
            eval_metric.reset()

def train(model_name: str, dataset, optimizer, batch_size=64, lr=0.01, epochs=100, device="cuda", momentum=0, plot=True):
    train, valid, test = get_dataloaders(dataset, model_name, batch_size)
    if dataset=="CIFAR10":
        image_size=32
    else:
        image_size=28
    model = create_network(model_name, image_size=image_size)
    model.to(device)
    
    #Optimizer is the string "adam" or "sgd"
    if optimizer=="adam":
        optimizer=torch.optim.Adam(model.parameters(), lr)
    elif optimizer=="sgd":
        optimizer=torch.optim.SGD(model.parameters(), lr, momentum)
    else:
        raise Exception("Optimizer should be adam or sgd")
    loss_fn = torch.nn.CrossEntropyLoss()
    
    datasets = {"valid": valid, "test": test, "train": train}
    epoch_stats = {"train_loss": [], "valid_loss": [], "test_loss": [], "train_acc": [], "valid_acc": [], "test_acc": []}
    
    record_metrics(model, epoch_stats, datasets, loss_fn)
    path = f"checkpoints/{model_name}_{dataset}_{lr}_0.pth"
    torch.save({"model": model, "epoch_stats": epoch_stats}, path)
    
    for epoch in range(1, epochs+1):
        model.train()

        for i, (x, y) in enumerate(train):
            optimizer.zero_grad()
            x, y= x.to(device), y.to(device, torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y) 
            loss.backward()
            optimizer.step()
        
        if epoch % 10 ==0:
            path = f"checkpoints/{model_name}_{dataset}_{lr}_{epoch}.pth"
            record_metrics(model, epoch_stats, datasets, loss_fn)
            torch.save({"model": model, "epoch_stats": epoch_stats}, path)
    return epoch_stats
# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

# TODO: Put functions related to pruning here

#%%
