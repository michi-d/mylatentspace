
"""Streamlit WebApp to visualize VAE latent spaces
"""

__author__ = "Michael Drews"
__copyright__ = "Copyright 2020, Michael Drews"
__email__ = "michaelsdrews@gmail.com"


import numpy as np
import streamlit as st
import time
import torch
from models import *

# set style
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


# load model
checkpoint = torch.load('checkpoints/model_10_epochs.pth.tar')
model = BasicVAE([256,128], 2)
model.load_state_dict(checkpoint['state_dict'])

'''
## Variational Autoencoder on Fashion MNIST
##
'''

'''
Explore the latent space interactively:
'''

x = st.slider('X', -1.0, 1.0, 0.0, format='%f')
y = st.slider('Y', -1.0, 1.0, 0.0, format='%f')

z = torch.Tensor([x,y])
sample = model.dec(z)
sample = sample.detach().numpy()[0, 0, :, :]

st.image(sample, width=200)

'''
####
#
Complete view of the latent space:
####
'''

N = 20

x = np.linspace(-1,1,N)
y = np.linspace(-1,1,N)
X,Y = np.meshgrid(x,y)

z = np.array(list(zip(X.flatten(), Y.flatten())))
z = torch.Tensor(z)
sample = model.dec(z)#.cuda()
    
image = torchvision.utils.make_grid(sample.reshape(N*N, 1, 28, 28), nrow=N)
image = image.detach().cpu().numpy().swapaxes(0,2).swapaxes(1,0)

st.image(image, width=600)
