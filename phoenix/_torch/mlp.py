import torch
import torch.nn as nn

class LearnableSquare(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.normal(0, 1, size=(1,)))
        self.b = nn.Parameter(torch.normal(0, 1, size=(1,)))
    
    def forward(self, x):
        return self.a*x*x + self.b*x

class MLP3(nn.Module):
    def __init__(self, input_sz, hidden_sz1, hidden_sz2, nb_classes):
        super().__init__()
        self.d1 = nn.Linear(input_sz, hidden_sz1, bias=True)
        self.act1 = LearnableSquare()
        self.d2 = nn.Linear(hidden_sz1, hidden_sz2, bias=True)
        self.act2 = LearnableSquare()
        self.d3 = nn.Linear(hidden_sz2, nb_classes, bias=True)
        
    def forward(self, x):
        h1 = self.act1(self.d1(x))
        h2 = self.act2(self.d2(h1))
        return self.d3(h2)

class MLP2(nn.Module):
    def __init__(self, input_sz, hidden_sz, nb_classes):
        super().__init__()
        self.d1 = nn.Linear(input_sz, hidden_sz, bias=True)
        self.act1 = LearnableSquare()
        self.d2 = nn.Linear(hidden_sz, nb_classes, bias=True)
        
    def forward(self, x):
        h1 = self.act1(self.d1(x))
        return self.d2(h1)