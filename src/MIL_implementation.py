import os
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math
import uuid
import random

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
from sklearn.datasets import make_moons, make_circles
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# need to add: padding and softmax mask for bags of variable size

class MIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_dim2: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        
        self.ff1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        self.attention_a = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim2),
            nn.Tanh()
        )
        
        self.attention_b = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim2),
            nn.Sigmoid()
        )
        
        self.projection = nn.Linear(self.hidden_dim2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n 1536
        Returns:
        - x: b 1 512
        """
        
        h = self.ff1(x)
        
        tanh_out = self.attention_a(h)
        sigmoid_out = self.attention_b(h)
        
        # elementwise product
        product = tanh_out * sigmoid_out
        
        score_projection = self.projection(product)
        
        attention_weights = torch.softmax(score_projection, dim=1)
        # b n 1
        
        # b n 
        
        # use attention weights to compute weighted sum of h
        return attention_weights, torch.einsum('bni,bnj->bij', attention_weights, h)
        
        
        
def test_mil():
    b = 2
    n = 3
    input_dim = 1536
    hidden_dim = 512
    hidden_dim2 = 256
    
    x = torch.randn(b, n, input_dim)
    
    mil = MIL(input_dim, hidden_dim, hidden_dim2)
    attention_weights, output = mil(x)
    
    print(attention_weights)
    print(output)
    
test_mil()