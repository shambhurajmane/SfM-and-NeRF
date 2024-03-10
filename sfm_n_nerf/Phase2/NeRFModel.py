import torch
import torch.nn as nn
import numpy as np
import ipdb

class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L=10, embed_direction_L=4, hidden_layers=256):
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
            nn.Linear(hidden_layers,  hidden_layers),
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
        
        # Repeat the input for batch processing
        batch_size = pos.shape[0]
        emb_x = torch.zeros((batch_size, pos.shape[1], self.embed_pos_L * 6 + 3))  
        emb_dir = torch.zeros((batch_size, direction.shape[1], self.embed_direction_L * 6 + 3)) 
        for i in range(pos.shape[0]):
            emb_x[i] = self.position_encoding(pos[i], self.embed_pos_L)
            emb_dir[i] = self.position_encoding(direction[i], self.embed_direction_L)
        emb_x = emb_x.to(pos.device)
        emb_dir = emb_dir.to(pos.device)    
        
        x = self.block1(emb_x)
        x = self.block2(torch.cat([x, emb_x], dim=-1))

        # Reshape x back to the original shape


        _, sigma = x[:, :,:-1], self.relu(x[:,:, -1])
        x = self.block3(torch.cat([x, emb_dir], dim=-1))
        output = self.block4(x)

        return output, sigma
