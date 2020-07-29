import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
import torchvision

class BasicEncoder(nn.Module):
    """Encoder for the basic model (two dense layers)
    """
    def __init__(self, n_channels=[8,4], n_latent_dims=2):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_latent_dims = n_latent_dims
        self.network, self.mu, self.log_var = self._create_network(n_channels, n_latent_dims)
        
    def _create_network(self, n_channels, n_latent_dims=2):
        
        network = nn.Sequential(nn.Linear(784, n_channels[0]),
                                nn.ReLU(),
                                nn.Linear(n_channels[0], n_channels[1]),
                                nn.ReLU()
        )
        
        mu = nn.Linear(in_features=n_channels[1], out_features=n_latent_dims)
        log_var = nn.Sequential(nn.Linear(in_features=n_channels[1], out_features=n_latent_dims),
                                nn.Softplus())  
        
        return network, mu, log_var
    
    def forward(self, x):
        
        x_enc = self.network(x.view(-1, 28*28))
        mu = self.mu(x_enc)
        log_var = self.log_var(x_enc)
        
        epsilon = torch.randn_like(log_var)
        z = mu + epsilon*log_var
        
        return z, mu, log_var
    
    
class BasicDecoder(nn.Module):
    """Decoder for the basic model (two dense layers)
    """
    
    def __init__(self, n_channels=[4,8], n_latent_dims=2):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_latent_dims = n_latent_dims
        self.network = self._create_network(n_channels, n_latent_dims)
        
    def _create_network(self, n_channels=[4,8], n_latent_dims=2):
        
        network = nn.Sequential(nn.Linear(n_latent_dims, n_channels[0]),
                                nn.ReLU(),
                                nn.Linear(n_channels[0], n_channels[1]),
                                nn.ReLU(),
                                nn.Linear(n_channels[1], 784),
                                nn.Sigmoid()
        )
        
        return network
    
    def forward(self, z):
        out = self.network(z).reshape(-1, 1, 28, 28)
        return out
    
    
class BasicVAE(nn.Module):

    def __init__(self, n_channels=[8, 4], n_latent_dims=2):
        super().__init__()
        
        self.enc = BasicEncoder(n_channels, n_latent_dims)
        self.dec = BasicDecoder(n_channels[::-1], n_latent_dims)
        
    def forward(self, x):
        
        z, mu, log_var = self.enc(x)
        out = self.dec(z)
        return out, mu, log_var, z