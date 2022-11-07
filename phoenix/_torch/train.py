import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import sys

from mlp import MLP2, MLP3

"""
    Trains an MLP on a dataset and saves the weights csv to be used by phoenix 
    Usage: "python3 train.py <mode> <dataset> <sigma>"
    --> <mode> is in {train, save_data}
    --> <dataset> is in {mnist, cifar} (for adult run train_adult.py) 
    --> sigma is the noise (only for train)
"""

def save_data(dataset, test_dataset):
    idx = 0
    with open(f'../data/{dataset}_test_10k.csv', 'w') as f:
        for i in range(test_dataset.data.shape[0]):
            dat = test_dataset.data[i]
            if dataset == 'cifar':
                dat = dat.transpose(2, 0, 1)
            l = dat.ravel().tolist()
            s = ','.join([f'{x}' for x in l])
            f.write(f'input{idx},{s}\n')
            tgt = test_dataset.targets[i]
            if dataset == 'mnist':
                tgt = tgt.cpu().item()
            f.write(f'target{idx},{tgt}\n')
            idx += 1
    print('wrote to the csv, exiting')


def main(mode, dataset, sigma):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_sz = 256
    nb_epochs = 100
    lr = 1e-2
    wd = 1e-3
    sched_steps = 25
    sched_gamma = 0.1
    criterion = nn.CrossEntropyLoss()
    nb_classes = 10 
    outs = nb_classes

    if dataset == 'mnist':
        input_sz = 28*28
        nb_classes = 10

        train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)
    elif dataset == 'cifar':
        input_sz = 32 * 32 * 3 
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)
    else:
        raise RuntimeError(f'Unknown dataset: {dataset}')

    if mode == 'save_data':
        save_data(dataset, test_dataset)
        exit()

    if dataset == 'mnist':
        model = MLP2(input_sz, 32, outs).to(device)
    else:
        model = MLP3(input_sz, 256, 32, outs).to(device)

    if dataset == 'cifar':
        mats = {
            'w1': model.d1.weight,
            'b1': model.d1.bias,
            'act1_a': model.act1.a,
            'act1_b': model.act1.b,
            'w2': model.d2.weight,
            'b2': model.d2.bias,
            'act2_a': model.act2.a,
            'act2_b': model.act2.b,
            'w3': model.d3.weight,
            'b3': model.d3.bias,
        }
    else:
        mats = {
            'w1': model.d1.weight,
            'b1': model.d1.bias,
            'act1_a': model.act1.a,
            'act1_b': model.act1.b,
            'w2': model.d2.weight,
            'b2': model.d2.bias
        }
    
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=sched_steps, gamma=sched_gamma)

    def accuracy(loader):
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(-1, input_sz)
            outs = model(inputs)
            outs = outs.view(-1, nb_classes)
            _, preds = torch.max(outs, 1)
            total += targets.shape[0]
            correct += (preds == targets).sum()

        acc = 100 * correct / total
        return acc

    # Train
    for epoch in range(nb_epochs):
        sum_loss = 0
        nb_examples = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            if sigma > 0:
                # uniform [-sigma, sigma]
                # inputs += torch.rand_like(inputs, device='cuda') * 2 * sigma - sigma 

                # gaussian
                inputs += torch.randn_like(inputs, device='cuda') * sigma
            else:
                pass

            inputs = inputs.view(-1, input_sz)

            opt.zero_grad()
            outs = model(inputs).view(-1, nb_classes)
            loss = criterion(outs, targets)
            loss.backward()
            
            sum_loss += loss
            nb_examples += inputs.shape[0]

            opt.step()
            if i%100 == 0:
                print("Epoch {}, avg epoch loss: {}".format(epoch + 1, sum_loss / nb_examples))
        
        # Test
        train_acc = accuracy(train_loader)
        test_acc = accuracy(test_loader)


        print(f'Accuracy | train: {train_acc:.2f} | test: {test_acc:.2f}')
        print()
        scheduler.step()

    # Save checkpoint as .pt
    # Save weights as csv for Phoenix
    #torch.save(model.state_dict(), f'out/{dataset}.pt')

    with open(f'../weights/{dataset}.csv', 'w') as f:
        for k, v in mats.items():
            l = v.ravel().detach().cpu().numpy().tolist()
            s = ','.join([f'{x:.32f}' for x in l])
            f.write(f'{k},{s}\n')

    print('saved, done')


#####################################
if __name__ == "__main__":
    mode = sys.argv[1]
    dataset = sys.argv[2]
    if len(sys.argv) > 3:
        sigma = float(sys.argv[3])
    else:
        sigma = None
    main(mode, dataset, sigma=sigma)