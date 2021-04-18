import torch.nn as nn

from bird_code import BIRD_CODE

"""
作成するモデル
"""

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.avg_pool = nn.AvgPool2d(kernel_size = (7, 18))
        self.linear = nn.Linear(1000, len(BIRD_CODE))
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        
        return x