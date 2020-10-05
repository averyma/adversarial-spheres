import torch.nn as nn

class quad(nn.Module):
    """
    Implementation of the "quadratic network" used in the 
    adversarial sphere paper: https://arxiv.org/abs/1801.02774
    """
    def __init__(self):
        super(quad, self).__init__()
        self.layer1 = nn.Linear(500, 500, bias = False)
        self.readout = nn.Linear(1, 1, bias = True)

    def act(self, x):
        return x**2
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = x.sum(dim = 1, keepdim = True)
        x = self.readout(x)
    
        return x.squeeze()
