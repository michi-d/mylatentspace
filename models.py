import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
import torchvision

class VanillaEncoder(nn.Module):
    """Encoder for the basic model (two dense layers)
    """
    def __init__(self, n_channels=[8,4], n_latent_dims=2):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_latent_dims = n_latent_dims
        self.network, self.mu, self.sigma = self._create_network(n_channels, n_latent_dims)
        
    def _create_network(self, n_channels, n_latent_dims=2):
        
        network = nn.Sequential(nn.Linear(784, n_channels[0]),
                                nn.ReLU(),
                                nn.Linear(n_channels[0], n_channels[1]),
                                nn.ReLU()
        )
        
        mu = nn.Linear(in_features=n_channels[1], out_features=n_latent_dims)
        sigma = nn.Sequential(nn.Linear(in_features=n_channels[1], out_features=n_latent_dims),
                                nn.Softplus())  
        
        return network, mu, sigma
    
    def forward(self, x):
        
        x_enc = self.network(x.view(-1, 28*28))
        mu = self.mu(x_enc)
        sigma = self.sigma(x_enc)
        
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon*sigma
        
        return z, mu, sigma
    
    
class VanillaDecoder(nn.Module):
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
    
    
class VAE(nn.Module):
    """Main class for a variational autoencoder
    """
    def __init__(self, layout, n_latent_dims):
        super().__init__()
        self.layout = layout
        self.n_latent_dims = n_latent_dims
       
    @classmethod  
    def construct_VanillaVAE(cls, layout=[8,4], n_latent_dims=2):
        """Constructor method for a Basic "vanilla" VAE
        """
        model = cls(layout, n_latent_dims)
        model.enc = VanillaEncoder(layout, model.n_latent_dims)
        model.dec = VanillaDecoder(layout[::-1], model.n_latent_dims)
        return model
        
    def forward(self, x):
        """Standard forward pass for all architectures
        """
        z, mu, sigma = self.enc(x)
        out = self.dec(z)
        return out, mu, sigma, z
    
    def get_data(self, name='fashionMNIST', batch_size=100, test_size=0.05):
        """Load the dataset and preprocess
        """
        
        if name == 'fashionMNIST':
            self.dataset = torchvision.datasets.FashionMNIST(root='./data', 
                                      train=True, download=True, 
                                      transform=torchvision.transforms.ToTensor())
            
        # train test split
        self._split_data(batch_size, test_size)
        
    def _split_data(self, batch_size=100, test_size=0.05):
        """Get data loaders for train and test set and define batch size
        """
        train_indices, test_indices = train_test_split(range(len(self.dataset)), 
                                                       test_size=test_size)

        # create Sampler objects
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # create DataLoader objects, passing batch_size and Samplers
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                           sampler=train_sampler)

        self.test_loader = torch.utils.data.DataLoader(self.dataset, sampler=test_sampler, 
                                           batch_size=batch_size)

        self.n_train = len(train_sampler.indices)
        self.n_test = len(test_sampler.indices)
        
        