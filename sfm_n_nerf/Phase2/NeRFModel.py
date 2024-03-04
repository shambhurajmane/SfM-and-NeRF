import torch
import torch.nn as nn
import numpy as np
import ipdb

class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, embed_direction_L=4, hidden_layers=128):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        self.embed_pos_L = embed_pos_L
        self.embed_direction_L = embed_direction_L
        
        self.block1 = nn.Sequential(
            nn.Linear(embed_pos_L * 6 + 3, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
        )   
        self.block2 = nn.Sequential(
            nn.Linear(embed_pos_L * 6 + 3 + hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers,  hidden_layers + 1),
        )
        self.block3 = nn.Sequential(
            nn.Linear(embed_direction_L * 6 + 3 + hidden_layers, hidden_layers//2),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_layers//2, 3),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()   
        
    @staticmethod   
    def position_encoding(x, L):
        #############################
        # Implement position encoding here
        #############################
        out = [x.type(torch.float32)]
        for i in range(0, L):
            out.append(torch.sin(2**i * x).type(torch.float32))
            out.append(torch.cos(2**i * x).type(torch.float32)) 
        y = torch.cat(out, dim=1)

        return y

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################
        
        emb_x= self.position_encoding(pos, self.embed_pos_L)
        emb_dir = self.position_encoding(direction, self.embed_direction_L)
        x = self.block1(emb_x)
        x = self.block2(torch.cat([x, emb_x], dim=1))
        
        x, sigma = x[:,:-1], self.relu(x[:,-1])
        x = self.block3(torch.cat([x, emb_dir], dim=1))
        output = self.block4(x)

        return output, sigma
