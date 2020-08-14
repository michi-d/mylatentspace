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
    
    """
    Main class for variational autoencoders.
    """
    
    def __init__(self, layout, n_latent_dims):
        super().__init__()
        self.layout = layout
        self.n_latent_dims = n_latent_dims
        self.loss_weights = [1, 1]
       
    @classmethod  
    def construct_VanillaVAE(cls, layout=[8,4], n_latent_dims=2):
        """Constructor method for a Basic "vanilla" VAE
        
        Args:
            layout: sub-class specific layout parameters
            n_latent_dims: number of dimensions in latent space
            
        Returns:
            model: VAE model instance
        """
        model = cls(layout, n_latent_dims)
        model.enc = VanillaEncoder(layout, model.n_latent_dims)
        model.dec = VanillaDecoder(layout[::-1], model.n_latent_dims)
        model.vae_type = 'Vanilla VAE'
        return model

    def set_optimizer(self, optimizer, learning_rate):
        """Set Optimizer and learning rate
        
        Args:
            optimizer: torch-Optimizer object
            learning_rate: learning_rate
        """
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.parameters(), lr=self.learning_rate)
        
    def set_loss_weights(self, loss_weights):
        """Set relative weight of reconstruction error and KL divergence
        """
        assert len(loss_weights) == 2
        self.loss_weights = loss_weights
        
    def forward(self, x):
        """Standard forward pass for all architectures
        """
        z, mu, sigma = self.enc(x)
        out = self.dec(z)
        return out, mu, sigma, z
    
    def loss_function(self, recon_x, x, mu, sigma):
        """Standard loss function for all architectures:
        Reconstruction error + KL divergence
        """
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        #KLD = 0.5*(mu**2 + log_var**2 - torch.log(log_var**2) - 1).sum()
        
        return self.loss_weights[0]*BCE + self.loss_weights[1]*KLD
    
    def get_data(self, name='fashionMNIST', batch_size=100, test_size=0.05):
        """Load the dataset and preprocess
        
        Args:
            name: Name of the dataset to load
            batch_size: batch size
            test_size: size of test set
        """
        
        if name == 'fashionMNIST':
            self.dataset = torchvision.datasets.FashionMNIST(root='./data', 
                                      train=True, download=True, 
                                      transform=torchvision.transforms.ToTensor())
            
        # train test split
        self._split_data(batch_size, test_size)
        
    def _split_data(self, batch_size=100, test_size=0.05):
        """Get data loaders for train and test set and define batch size
        
        Args:
            batch_size: batch size
            test_size: size of test set
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
        
    def print_hyperparameters(self):
        """Print hyperparameters"""
        print(f'VAE type: {self.vae_type}')
        print(f'layout: {self.layout}')
        print(f'n_latent_dims: {self.n_latent_dims}')
        print(f'batch size: {self.train_loader.batch_size}')
        #print(f'n_samples train: {self.n_train}')
        #print(f'n_samples test: {self.n_test}')
        print(f'Optimizer: {type(self.optimizer)}')
        print(f'Learning rate: {self.learning_rate}')
        print(f'Loss weights:{self.loss_weights}')
        
    def train_one_epoch(self, epoch, verbose=False):
        """Do one training epoch
        """
        self.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            #data = data.cuda()
            self.optimizer.zero_grad()

            recon_batch, mu, sigma, z = self(data)
            loss = self.loss_function(recon_batch, data, mu, sigma)

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            if verbose:
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), self.n_train,
                        100. * batch_idx / self.n_train, loss.item() / len(data)))
                
        if verbose:
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / self.n_train))
                    
        return train_loss
    
    def evaluate(self, verbose=False):
        """Evaluate model on test set
        """
        self.eval()
        test_loss= 0
        with torch.no_grad():
            for data, _ in self.test_loader:
                #data = data.cuda()
                recon, mu, sigma, z = self(data)

                # sum up batch loss
                test_loss += self.loss_function(recon, data, mu, sigma).item()

        test_loss = test_loss/self.n_test
        
        if verbose:
            print('====> Test set loss: {:.4f}'.format(test_loss))
            
        return test_loss

    def fit(self, epochs, verbose=False):
        """Fit model
        
        Args: 
            epochs: number of epochs
            verbose: whether to show print statements or not
        """
        for epoch in range(1, epochs+1):
            train_loss = self.train_one_epoch(epoch, verbose=verbose)
            test_loss = self.evaluate(verbose)
        
    def get_training_batch(self):
        """Get one batch from training data
        """
        images, labels = next(iter(model.train_loader))
        return images, labels
    
    def run(self, data):
        """Run model on data in eval mode
        """
        model.eval()
        recon, mu, sigma, z = model(data)
        return recon.detach().cpu(), mu.detach().cpu(), sigma.detach().cpu(), z.detach().cpu()

    def get_grid_reconstructions(self, N=10, nrow=8):
        """Generate grid showing reconstructed image alongside originals.

        Args:
            recon: recontructed images
            orig: original images
            N: number of pairs
            nrow: number of rows in the grid

        Returns:
            grid: np.array ready for use in plt.imshow
        """

        images, labels = self.get_training_batch()
        recon, mu, sigma, z = self.run(images)

        if N > recon.shape[0]:
            print(f'Too many images requested, set N to {recon.shape[0]}')
            N = recon.shape[0]

        img = np.vstack(list(zip(images[:N, :].numpy(), recon[:N, :].numpy())))
        img = torch.Tensor(img.reshape(2*N, 1, 28, 28))
        grid = torchvision.utils.make_grid(img, nrow=nrow)
        grid = grid.numpy().swapaxes(0,2).swapaxes(1,0)
        return grid
    
    def random_sample_latent_space(self, n_sample=100):
        """Generate random vectors in latent space and decode.
        
        Args:
            n_sample: number of samples to generate
            
        Returns:
            grid: image in np.array format
        """
        
        with torch.no_grad():
            z = torch.randn(n_sample, 2)
            sample = self.dec(z)
            
        grid = torchvision.utils.make_grid(sample.reshape(n_sample, 1, 28, 28))\
                                           .detach().cpu().numpy().swapaxes(0,2).swapaxes(1,0)

        return grid
    
    def generate_2D_grid_latent_space(self, N=20):
        """Generate a grid of vectors in 2D latent space and decode.
        
        Args:
            N: number of images per row/column
            
        Returns:
            grid: image in np.array format
        """
        
        assert self.n_latent_dims == 2
        x = np.linspace(-1,1,N)
        y = np.linspace(-1,1,N)
        X,Y = np.meshgrid(x,y)

        with torch.no_grad():
            z = np.array(list(zip(X.flatten(), Y.flatten())))
            z = torch.Tensor(z)
            sample = self.dec(z)#.cuda()

        img = torchvision.utils.make_grid(sample.reshape(N*N, 1, 28, 28), nrow=N)\
                   .detach().cpu().numpy().swapaxes(0,2).swapaxes(1,0)
            
        return img

