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

def accuracy(model, epoch_stats, datasets, loss_fn):
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

def simple_train(model_name, dataset, optimizer, batch_size=64, lr=0.01, epochs=100, device="cuda", momentum=0, plot=True):
    train, valid, test = get_dataloaders(dataset, batch_size)
    image_size = int(len(train.dataset.getitem(0)[0])**(1/2))
    model = create_network(model_name, image_size=image_size)
    model.to(device)
    
    #Optimizer is the string "adam" or "sgd"
    if optimizer=="adam":
        optimizer=torch.optim.Adam(model.parameters(), lr)
    elif optimizer=="sgd":
        optimizer=torch.optim.SGD(model.parameters(), lr, momentum)
    else:
        return Exception("Optimizer should be adam or sgd")
    loss_fn = torch.nn.CrossEntropyLoss()

    #path = f"checkpoints/{model_name}_{dataset}_{lr}_0.pth"
    #torch.save(model, path)
    
    datasets = {"valid": valid, "test": test, "train": train}
    
    epoch_stats = {"train_loss": [], "valid_loss": [], "test_loss": [], "train_acc": [], "valid_acc": [], "test_acc": []}
    record_metrics(model, epoch_stats, datasets, loss_fn)

    if plot:
        animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs], figsize=(10, 5),
                                legend=['train loss', 'train accuracy', 'test loss', 'test accuracy'])
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
            #path = f"checkpoints/{model_name}_{dataset}_{lr}_{epoch}.pth"
            record_metrics(model, epoch_stats, datasets, loss_fn)
            #torch.save({"model": model, "epoch_stats": epoch_stats}, path)
            if plot:
                animator.add(epoch + 1, (epoch_stats["train_loss"][-1], epoch_stats["train_acc"][-1], epoch_stats["valid_loss"][-1], epoch_stats["valid_acc"][-1], epoch_stats["test_loss"][-1], epoch_stats["test_acc"][-1]))
    return epoch_stats

#%%
