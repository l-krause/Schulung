import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Neuronales Netz mit 1 Input Layer, 1 Hidden Layer und 1 Output Layer. 
"""
class NeuralNet(nn.Module):
        
    def __init__(self, in_features: int, hidden_features: int, out_features: int, device = "cpu"):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features, device=device, bias=True)
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features, device=device, bias=True)

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        pred = self.fc1(x)
        pred = F.relu(pred)
        pred = self.fc2(pred)
        return pred
