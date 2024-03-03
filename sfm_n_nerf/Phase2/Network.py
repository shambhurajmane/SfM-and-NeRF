import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class NeRF_NW():
    super.__init__(self,input_channels, width)
    self.fc1 = nn.Sequential(nn.Linear(input_channels, width, dtype=torch.double), nn.ReLU())
    self.fc2 = nn.Sequential(nn.Linear(width, width, dtype=torch.double), nn.ReLU())
    self.fc3 = nn.Sequential(nn.Linear(width + input_channels, width, dtype=torch.double), nn.ReLU())
    self.fc4 = nn.Sequential(nn.Linear(width, 4, dtype=torch.double))

    def forward(x):        
        x = self.fc1(x)        
        x = self.fc2(x)
        x = self.fc3(x)
        return x