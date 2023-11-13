# This is a file where you should put your own functions
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


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

# TODO: Datasets go here.

def get_dataloaders(dataset, model_name="lenet", batch_size=60):
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
        train_dataset = torchvision.datasets.FashionMNIST("data/fashionmnist", train=True, download=True,
                                                          transform=transform)
        test_data = torchvision.datasets.FashionMNIST("data/fashionmnist", train=False, download=True,
                                                      transform=transform)
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    valid_size = 5000
    train_size = len(train_dataset) - valid_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
    test = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
    return train, valid, test


def get_image_size(dataset_name):
    if dataset_name in ["MNIST", "FashionMNIST"]:
        return 28
    elif dataset_name == "CIFAR10":
        return 32
    else:
        raise Exception(f"Unknown dataset: {dataset_name}")


# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------

# TODO: Define network architectures here

def create_lenet(image_size=28):
    return torch.nn.Sequential(
        torch.nn.Linear(int(image_size ** 2), 300),
        torch.nn.ReLU(),
        torch.nn.Linear(300, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    )


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding='same'):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=1)
        self.batch1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batch2 = torch.nn.BatchNorm2d(out_channels)

        self.skip_connection = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides,
                                               padding=0)
        self.batch_res = torch.nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = torch.nn.functional.relu(self.batch1(self.conv1(X)))
        Y = self.batch2(self.conv2(Y))
        Y += self.batch_res(self.skip_connection(X))
        return torch.nn.functional.relu(Y)


def create_resnet_18(image_size=32):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1, stride=2), torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

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
        torch.nn.Flatten(),
        torch.nn.Linear(int((image_size / 32) ** 2 * 64), 10),
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
        torch.nn.Flatten(),
        torch.nn.Linear(int((image_size / 32) ** 2 * 512), 10),
    )


def create_network(arch, **kwargs):
    # TODO: Change this function for the architectures you want to support
    if arch == 'lenet':
        return create_lenet(**kwargs)
    elif arch == 'resnet18':
        return create_resnet_18(**kwargs)
    elif arch == 'vgg19':
        return create_vgg_19(**kwargs)
    else:
        raise Exception(f"Unknown architecture: {arch}")


def get_num_epochs(net_name):  # values taken from figure2, batch size and the data dimensions (paper_iter / (len/60))
    return {"lenet": 55, "resnet18": 70, "vgg19": 130}[net_name]


def get_lr_optimizer(net_name):
    return {"lenet": (1.2e-3, "adam", 0), "resnet18": (1e-3, "sgd", 0.9), "vgg19": (1e-3, "adam", 0.9)}[net_name]


# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

def load_network(experiment_name, lr, optimizer, momentum=0, save_path="checkpoints", device=d2l.try_gpu()):
    path = os.path.join(os.getcwd(), save_path, f"{experiment_name}_{lr}_{optimizer}_{momentum}")
    model = torch.load(os.path.join(path, "final.pth"), map_location=device)
    epoch_stats = pd.read_csv(os.path.join(path, "epoch_stats.csv"), index_col="epoch").to_dict(orient="list")
    return model, epoch_stats


def resume_training(model, path, epochs):
    epoch_stats = {"train_loss": [], "valid_loss": [], "test_loss": [], "train_acc": [], "valid_acc": [],
                   "test_acc": []}

    if not os.path.exists(path) or os.listdir(path) == []:
        if not os.path.exists(path):
            os.mkdir(path)
        return model, epoch_stats, 1

    epoch_stats_df = pd.read_csv(os.path.join(path, "epoch_stats.csv"), index_col="epoch")
    last_epoch = epoch_stats_df.index[-1]

    if last_epoch == epochs:
        model = torch.load(os.path.join(path, "final.pth"))
    else:
        model.load_state_dict(torch.load(os.path.join(path, f"{last_epoch}.pth"))["model_state_dict"])

    epoch_stats = epoch_stats_df.to_dict(orient="list")
    return model, epoch_stats, last_epoch + 1


def save_training(model, path, epoch_stats, epoch):
    checkpoint_path = os.path.join(path, f"{epoch}.pth")
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    epoch_stats = pd.DataFrame(epoch_stats, index=range(1, len(epoch_stats["train_loss"]) + 1))
    epoch_stats.index.name = "epoch"
    epoch_stats.to_csv(os.path.join(path, "epoch_stats.csv"))


def record_metrics(model, epoch_stats, datasets, loss_fn, device):
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


def train(net, net_type, datasets, experiment_name, optimizer="adam", lr=0.01, epochs=100, save_path="checkpoints",
          device=d2l.try_gpu(), momentum=0, plot=True, save_patience=2, early_stop_metric=None):
    path = os.path.join(os.getcwd(), save_path, f"{experiment_name}_{lr}_{optimizer}_{momentum}")

    model = create_network(net_type)
    prune_network_from_mask(net, model)
    model.load_state_dict(net.state_dict())

    animator = None
    model.to(device)

    if early_stop_metric not in ["valid_acc", "valid_loss", "train_loss", "test_loss", None]:
        raise Exception("Invalid early stop metric")

    # Optimizer is the string "adam" or "sgd"
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum)
    else:
        raise Exception("Optimizer should be adam or sgd")
    loss_fn = torch.nn.CrossEntropyLoss()

    model, epoch_stats, start_epoch = resume_training(model, path, epochs)

    if start_epoch > epochs:
        return model, epoch_stats

    if plot:
        animator = d2l.Animator(xlabel='epoch', xlim=[start_epoch, epochs], figsize=(10, 5),
                                legend=['train loss', 'train accuracy', "valid_loss", "valid_acc", 'test loss',
                                        'test accuracy'])
    final_model = model
    for epoch in range(start_epoch, epochs + 1):
        model.train()

        for i, (x, y) in enumerate(datasets['train']):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device, torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

        record_metrics(model, epoch_stats, datasets, loss_fn, device)

        if epoch % save_patience == 0:
            save_training(model, path, epoch_stats, epoch)

        if plot:
            animator.add(epoch, (
                epoch_stats["train_loss"][-1], epoch_stats["train_acc"][-1], epoch_stats["valid_loss"][-1],
                epoch_stats["valid_acc"][-1], epoch_stats["test_loss"][-1], epoch_stats["test_acc"][-1]))
        else:
            print(f"Epoch: {epoch}", ": ", end="")
            print_training_results_model(epoch_stats)

    if early_stop_metric is not None:
        # model.load_state_dict(torch.load(os.path.join(path, f"{epoch-early_stop_patience}.pth")))
        best_epoch_find = np.argmax if early_stop_metric in ["valid_acc", "test_acc"] else np.argmin
        epoch_stats['early_stop_epoch'] = (best_epoch_find(epoch_stats[early_stop_metric]
                                                           [save_patience - 1::save_patience]) + 1) * save_patience
        final_model.load_state_dict(torch.load(os.path.join(path, f"{epoch_stats['early_stop_epoch']}.pth"))
                                    ["model_state_dict"])

    save_training(model, path, epoch_stats, epochs)  # save last epoch and eventual change to epoch_stats

    print_training_results_model(epoch_stats)
    torch.save(final_model, os.path.join(path, "final.pth"))
    return model, epoch_stats


def print_training_results_model(epoch_stats):
    epoch = -1 if "early_stop_epoch" not in epoch_stats else epoch_stats["early_stop_epoch"] - 1
    print(f"train loss {epoch_stats['train_loss'][epoch]:.3f}, train acc {epoch_stats['train_acc'][epoch]:.3f}, "
          f"valid loss {epoch_stats['valid_loss'][epoch]:.3f}, valid acc {epoch_stats['valid_acc'][epoch]:.3f}, "
          f"test loss {epoch_stats['test_loss'][epoch]:.3f}, test acc {epoch_stats['test_acc'][epoch]:.3f}")


def print_plot_results_model(epoch_stats, title):
    epoch_stats_to_plot = {
        key: value for key, value in epoch_stats.items() if key != 'early_stop_epoch'}
    pd.DataFrame(epoch_stats, index=range(1, len(epoch_stats['train_loss']) + 1)
                 ).plot(xlabel="epoch", ylabel="metric value", title=title, grid=True)

    if "early_stop_epoch" in epoch_stats:
        plt.axvline(epoch_stats["early_stop_epoch"][-1], color="red", linestyle="--")

    print(title, ": ", end="")
    print_training_results_model(epoch_stats)


# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

# TODO: Put functions related to pruning here
def local_prune(net, fc_amount, conv_amount, out_amount, prune_operation):
    """
    Note: conv_amount and out_amount are modifier to fc_amount
    """
    for layer in list(net.children())[:-1]:
        if isinstance(layer, torch.nn.Linear):
            prune_operation(layer, name="weight", amount=fc_amount)
        elif isinstance(layer, torch.nn.Conv2d):
            prune_operation(layer, name="weight", amount=fc_amount * conv_amount)
    prune_operation(list(net.children())[-1], name="weight", amount=fc_amount * out_amount)


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
    """
    Note: conv_amount and out_amount are modifier to amount
    """
    if net_type == "lenet":
        if conv_amount is None:
            conv_amount = 1
        if out_amount is None:
            out_amount = 0.5
        local_prune(net, amount, conv_amount, out_amount, prune_type.apply)
    else:
        global_prune(net, amount, prune_type)


def prune_network_from_mask(net_in, net_out):
    for layer_in, layer_out in zip(net_in.children(), net_out.children()):
        if prune.is_pruned(layer_in) and isinstance(layer_in, (torch.nn.Linear, torch.nn.Conv2d)):
            prune.custom_from_mask(layer_out, name="weight", mask=layer_in.weight_mask)
        elif prune.is_pruned(layer_in) and isinstance(layer_in, ResidualBlock):
            for sublayer_in, sublayer_out in zip(layer_in.children(), layer_out.children()):
                if isinstance(sublayer_in, (torch.nn.Linear, torch.nn.Conv2d)):
                    prune.custom_from_mask(sublayer_out, name="weight", mask=sublayer_in.weight_mask)


def get_pruned_params(net, params_before):
    params_after = params_before - sum(torch.sum(b == 0) for b in net.buffers()).item()
    return round(abs(1 - (params_after / params_before)), 4)


def random_pruning_training(net, net_type, datasets, experiment_folder_name, save_path="checkpoints",
                            pruning_targets=None, conv_amount=None, out_amount=None,
                            prune_type=prune.RandomUnstructured, **training_kwargs):
    if pruning_targets is None:
        pruning_targets = [0.2]
    elif type(pruning_targets) is not list:
        pruning_targets = [pruning_targets]
    path = os.path.join(os.getcwd(), save_path, f"{experiment_folder_name}")

    if not os.path.exists(path):
        os.mkdir(path)

    if training_kwargs.get("epochs") is None:
        training_kwargs["epochs"] = get_num_epochs(net_type)

    results_model = {}
    for pruning_amount in pruning_targets:
        net_name = "{:.2%}".format((1 - pruning_amount))[:-1].zfill(6)

        net_pruned = copy.deepcopy(net)
        prune_network(net_pruned, net_type, pruning_amount, prune_type=prune_type, conv_amount=conv_amount,
                      out_amount=out_amount)
        trained_net_pruned, stats = train(net_pruned, net_type, datasets, net_name, save_path=path, **training_kwargs)

        torch.save(trained_net_pruned, os.path.join(path, f"{net_name}.pth"))
        results_model[net_name] = trained_net_pruned

    return results_model


def one_shot_pruning_training(net, net_type, datasets, experiment_folder_name, save_path="checkpoints", retraining=True,
                              pruning_targets=None, conv_amount=None, out_amount=None, prune_type=prune.L1Unstructured,
                              **training_kwargs):
    if pruning_targets is None:
        pruning_targets = [0.2]
    elif type(pruning_targets) is not list:
        pruning_targets = [pruning_targets]
    path = os.path.join(os.getcwd(), save_path, f"{experiment_folder_name}")

    if not os.path.exists(path):
        os.mkdir(path)

    if training_kwargs.get("epochs") is None:
        training_kwargs["epochs"] = get_num_epochs(net_type)

    trained_net, stats = train(net, net_type, datasets, "100.00", save_path=path, **training_kwargs)

    results_model = {}
    for pruning_amount in pruning_targets:
        net_name = "{:.2%}".format((1 - pruning_amount))[:-1].zfill(6)
        net_pruned = copy.deepcopy(trained_net)
        prune_network(net_pruned, net_type, pruning_amount, prune_type=prune_type, conv_amount=conv_amount,
                      out_amount=out_amount)

        net_pruned_reinit = copy.deepcopy(net)
        prune_network_from_mask(net_pruned, net_pruned_reinit)
        torch.save(net_pruned_reinit, os.path.join(path, f"{net_name}.pth"))
        results_model[net_name] = net_pruned



        if retraining:
            trained_net_pruned, stats = train(net_pruned, net_type, datasets, f"trained_{net_name}", save_path=path,
                                              **training_kwargs)
            torch.save(trained_net_pruned, os.path.join(path, f"trained_{net_name}.pth"))
            results_model["trained_" + net_name] = trained_net_pruned

    return results_model


def iterative_pruning_training(net, net_type, datasets, experiment_folder_name, save_path="checkpoints", reinit=False,
                               pruning_amount=0.2, pruning_target=0.01, conv_amount=None, out_amount=None, reset=False,
                               prune_type=prune.L1Unstructured, **training_kwargs):
    path = os.path.join(os.getcwd(), save_path, f"{experiment_folder_name}")
    if not os.path.exists(path) or os.listdir(path) == []:
        if not os.path.exists(path):
            os.mkdir(path)

    if training_kwargs.get("epochs") is None:
        training_kwargs["epochs"] = get_num_epochs(net_type)

    total_params = sum(p.numel() for p in net.parameters())
    
    trained_net, stats = train(net, net_type, datasets, "100.00", save_path=path, **training_kwargs)
    prune_network(trained_net, net_type, pruning_amount, prune_type=prune_type, conv_amount=conv_amount,
                  out_amount=out_amount)

    if reset:
        net_tmp = copy.deepcopy(net)
        prune_network_from_mask(trained_net, net_tmp)
        trained_net = net_tmp

    pruned_params = get_pruned_params(trained_net, total_params)

    results_model, results_stats = {}, {}
    while (1 - pruned_params) > pruning_target:
        net_name = "{:.2%}".format((1 - pruned_params))[:-1].zfill(6)
        results_model[net_name] = trained_net
        # torch.save(trained_net, os.path.join(path, f"{net_name}.pth"))

        if reinit:
            net_tmp = create_network(net_type)
            prune_network_from_mask(trained_net, net_tmp)
            trained_reinit_net, stats = train(net_tmp, net_type, datasets, f"reinit_{net_name}", save_path=path,
                                              **training_kwargs)
            results_model["reinit_" + net_name] = trained_reinit_net
            save_epoch = -1 if "early_stop_epoch" not in stats.keys() else stats["early_stop_epoch"] - 1
            results_stats["reinit_" + net_name] = {key: value[save_epoch] for key, value in stats.items()}

        trained_net, stats = train(trained_net, net_type, datasets, f"trained_{net_name}", save_path=path, **training_kwargs)
        results_model["trained_" + net_name] = trained_net
        
        save_epoch = -1 if "early_stop_epoch" not in stats.keys() else stats["early_stop_epoch"] - 1
        results_stats["trained_" + net_name] = {key: value[save_epoch] for key, value in stats.items()}
        prune_network(trained_net, net_type, pruning_amount, prune_type=prune_type, conv_amount=conv_amount,
                      out_amount=out_amount)

        if reset:
            net_tmp = copy.deepcopy(net)
            prune_network_from_mask(trained_net, net_tmp)
            trained_net = net_tmp
        pruned_params = get_pruned_params(trained_net, total_params)

    if not reset:
        net_tmp = copy.deepcopy(net)
        prune_network_from_mask(trained_net, net_tmp)
        trained_net = net_tmp

    net_name = str(int((1 - pruned_params) * 100)).zfill(3)
    # torch.save(trained_net, os.path.join(path, f"{net_name}.pth"))
    results_model[net_name] = trained_net
    trained_net, stats = train(trained_net, net_type, datasets, f"trained_{net_name}", save_path=path, **training_kwargs)

    save_epoch = -1 if "early_stop_epoch" not in stats.keys() else stats["early_stop_epoch"] - 1
    results_stats["trained_" + net_name] = {key: value[save_epoch] for key, value in stats.items()}
    results_model["trained_" + net_name] = trained_net

    return results_model, results_stats
