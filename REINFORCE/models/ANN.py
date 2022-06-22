# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_V1(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_V1, self).__init__()
        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, outputs)
        self.head = nn.Softmax(dim=0)
        
        self.ups=1e-7

    def forward(self, x):
        x = x
        x = F.relu(self.fc1(x))
        x = self.head(self.fc2(x))
        
        return x
