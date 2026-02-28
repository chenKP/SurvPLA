import torch
from baseVAE import BaseVAE
from torch import nn
from torch.nn import functional as F
# from .types_ import *
from feature_fusion import FusionModal
from typing import List
class BetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 path_in_dim: int,
                 omics_in_dim: int,
                 out_dim: int,
                 latent_dim: int,
                 latent_embed_dim: int,
                 hidden_dims: List = None,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'H',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()
        self.path_in_dim = path_in_dim
        self.omics_in_dim = omics_in_dim
        self.latent_dim = latent_dim
        self.latent_embed_dim = latent_embed_dim
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        if hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 128]
        else:
            self.hidden_dims = hidden_dims

        #The code is being organized...

 
    def init_fusion(self, path_in_dim, omics_in_dim, out_dim):
        modules = []
        modules.append(
            nn.Sequential(FusionModal(path_in_dim, omics_in_dim, out_dim),
                          nn.LeakyReLU())
            
        )
        self.fusion = nn.Sequential(*modules)


    def init_decoder_path(self):
        modules = []
        #The code is being organized...
        self.decoder_path = nn.Sequential(*modules)
    def init_decoder_omics(self):
        modules = []
        #The code is being organized...
        self.decoder_omics = nn.Sequential(*modules)
    

        
    def encode_path(self, input):
        
        result = torch.flatten(input, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_path(result)
        log_var = self.fc_var_path(result)
        return [mu, log_var]
    def encode_omics(self, input):

        #The code is being organized...


    def decode(self, z):
        #The code is being organized...


    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, path, omics, **kwargs):
        
        #The code is being organized...


    def loss_function(self, path, omics, path_recons, omics_recons,mu_path,log_var_path,
                mu_omics,log_var_omics ,
                kld_weight = 0.00025):#0.00025
        self.num_iter += 1
        kld_weight = kld_weight  # Account for the minibatch samples from the dataset
        recons_loss_path =F.mse_loss(path, path_recons)
        recons_loss_omics =F.mse_loss(omics, omics_recons)
        recons_loss = recons_loss_path + recons_loss_omics

        kld_loss_omics = torch.mean(-0.5 * torch.sum(1 + log_var_omics - mu_omics ** 2 - log_var_omics.exp(), dim = 1), dim = 0)
        kld_loss_path = torch.mean(-0.5 * torch.sum(1 + log_var_path - mu_omics ** 2 - log_var_path.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss_omics + kld_loss_path

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss +  kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]