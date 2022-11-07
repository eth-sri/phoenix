from datasets.adult_full import AdultFullDataset
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from torch.distributions.multivariate_normal import MultivariateNormal
import argparse
from mlp import MLP2

################################
# Parse and set args

parser = argparse.ArgumentParser(description='Fairness')
parser.add_argument('--mode', type=str, default='train') 
parser.add_argument('--dataset', type=str, default='adult_full')
parser.add_argument('--sigma_scale', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--filename', type=str, default='../weights/adult.csv')
args = parser.parse_args()
    
# Seed and device 
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# params
model_name = f'adult'
batch_sz = 256
nb_epochs = 50
lr = 0.01 
nb_classes = 2

# Load the dataset 
train_dataset = AdultFullDataset('train', None)
val_dataset = AdultFullDataset('validation', None)
test_dataset = AdultFullDataset('test', None)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_sz, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)


# Uncomment to save data for Phoenix
"""
with open('../data/adult_test_15060.csv', 'w') as f:
    for i in range(test_dataset.features.shape[0]):
        l = test_dataset.features[i].ravel().cpu().numpy().tolist()
        s = ','.join([f'{x:.10f}' for x in l])
        f.write(f'input{i},{s}\n')
        f.write(f'target{i},{int(test_dataset.labels[i].cpu().item())}\n')
    print('test data written!')
"""

input_sz = train_dataset.features.shape[1]
cont_sz = 6

# Defining the similarity for fairness
# Cont features: age, fnlwgt, education_num, capital_gain, capital_loss, hrs_per_week
# "Assuming other features are fixed, how much should 2 SIMILAR people differ in age at most" -> (2)
sim_thresh = torch.FloatTensor([4, 10000, 1, 500, 50, 8]).to(device)
sim_thresh /= train_dataset.std # normalize

sigma = torch.diag(sim_thresh).to(device)
S = torch.diag(1.0 / torch.diag(sigma))
sqrtinvL = 1

print(f'Original sigma={sigma}\n S={S}\n sqrtinvL={sqrtinvL}')

# Normalize s.t. total variation is input_sz * scale
total_var = sim_thresh.sum().cpu().item()
scaling_factor = (cont_sz / total_var) * args.sigma_scale
sigma *= scaling_factor 
S = torch.diag(1.0 / torch.diag(sigma))
sqrtinvL = (1.0 / scaling_factor)
eigs, _ = torch.linalg.eigh(sigma)

print(f'Final sigma={sigma}\n S={S}\n sqrtinvL={sqrtinvL}')
print(f'Total variation of sigma: {eigs.sum().item()} (should be 6*sigma scale)')
print(f'(L={(1 / (sqrtinvL * sqrtinvL)):.3f})')

# Make a noise generator
noise_gen = MultivariateNormal(torch.zeros(sigma.shape[0]).to(device), sigma)

# Training
outs = 1
model = MLP2(input_sz, 32, outs).to(device)

# Training params
criterion = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, 'min', patience=5, factor=0.5
)

# Helper to get accuracy
def accuracy(loader):
    correct, total = 0, 0
    for inputs, targets, protected in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outs = model(inputs.view(-1, input_sz)).squeeze()
        preds = (outs > 0).int()
        total += targets.shape[0]
        correct += (preds == targets).sum()
    acc = 100 * correct.item() / total
    return acc

# Training loop
for epoch in range(nb_epochs):
    sum_loss = 0
    nb_examples = 0
    for i, (inputs, targets, protected) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # If not clean add noise to the input before passing
        noise = torch.stack([noise_gen.sample() for _ in range(inputs.shape[0])], dim=0)
        if input_sz > cont_sz:
            zeros = torch.zeros((noise.shape[0], input_sz-cont_sz)).to(device)
            noise = torch.cat([noise, zeros], dim=1)
        assert noise.shape == inputs.shape ###
        inputs += noise.to(device)

        opt.zero_grad()
        outs = model(inputs.view(-1, input_sz)).squeeze()
        loss = criterion(outs, targets)
        loss.backward()
        opt.step()
        
        sum_loss += loss
        nb_examples += inputs.shape[0]

    # Test in each epoch
    train_acc = accuracy(train_loader)
    val_acc = accuracy(val_loader)
    test_acc = accuracy(test_loader)
    if (epoch+1)%1 == 0:
        print(f'[Epoch {epoch+1}] | loss: {(sum_loss/nb_examples):.5f} | train: {train_acc:.3f} | val: {val_acc:.3f} | test: {test_acc:.3f}')
    scheduler.step(val_acc)

# Save
#torch.save(model.state_dict(), f'out/{model_name}.pt')
mats = {
        'w1': model.d1.weight,
        'b1': model.d1.bias,
        'act1_a': model.act1.a,
        'act1_b': model.act1.b,
        'w2': model.d2.weight,
        'b2': model.d2.bias
    }
with open(f'../weights/adult.csv', 'w') as f:
    for k, v in mats.items():
        l = v.ravel().detach().cpu().numpy().tolist()
        s = ','.join([f'{x:.32f}' for x in l])
        f.write(f'{k},{s}\n')