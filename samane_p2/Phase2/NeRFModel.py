import torch
import torch.nn as nn
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################

    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################

        return y

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################

        return output
