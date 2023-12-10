import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

import logging
import numpy as np
import copy
import pickle


_logger = logging.getLogger(__name__)

cuda = True if torch.cuda.is_available() else False


class MultiCategoryGumbelSoftmax(torch.nn.Module):
    """Gumbel softmax for multiple output categories

    Parameters
    ----------
    input_dim : int
        Dimension for input layer
    output_dims : list of int
        Dimensions of categorical output variables
    tau : float
        Temperature for Gumbel softmax
    """
    def __init__(self, input_dim, output_dims, tau=1/2):
        super(MultiCategoryGumbelSoftmax, self).__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(input_dim, output_dim)
            for output_dim in output_dims
        )
        self.tau = tau

    def forward(self, x):
        xs = tuple(layer(x) for layer in self.layers)
        logits = tuple(F.log_softmax(x, dim=1) for x in xs)
        categorical_outputs = tuple(
            F.gumbel_softmax(logit, tau=self.tau, hard=True, eps=1e-10)
            for logit in logits
        )
        return torch.cat(categorical_outputs, 1)


# 定义共享层和生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dims,multi_branch_dim_list):
        super(Generator, self).__init__()

        self.shared_layer = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.branches = nn.ModuleList()
        for i, dim in enumerate(output_dims):
            branch = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                MultiCategoryGumbelSoftmax(128, multi_branch_dim_list[i])
            )
            self.branches.append(branch)

    def forward(self, z, i):
        shared_out = self.shared_layer(z)
        # outputs = []
        output = self.branches[i](shared_out)
        # for branch in self.branches:
        #     output = branch(shared_out)
        #     outputs.append(output)
        return output

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Verti_GAN(object):
    """Class to store, train, and generate from a
    differentially-private Wasserstein GAN

    Parameters
    ----------
    generator : torch.nn.Module
        torch Module mapping from random input to synthetic data

    discriminator : torch.nn.Module
        torch Module mapping from data to a real value

    noise_function : function
        Mapping from number of samples to a tensor with n samples of random
        data for input to the generator. The dimensions of the output noise
        must match the input dimensions of the generator.
    """
    def __init__(self, global_generator, local_discriminators, noise_function,party2attr):
        self.generator = global_generator
        self.discriminator = local_discriminators
        self.noise_function = noise_function
        self.party2attr = party2attr
        self.epsilon2sigma = {0.4:2.1,0.8:1.2,1.6:0.88,3.2:0.7,None:0}
        #For NLTCS dataset
        # (0.4, sigma=2.1, batch_size/len(data)=100/23574, clipping_bound = 1, iter=10, party=2)
        # (0.8, sigma=1.2, batch_size/len(data)=100/23574, clipping_bound = 1, iter=10, party=2)
        # (1.6, sigma=0.88, batch_size/len(data)=100/23574, clipping_bound = 1, iter=10, party=2)
        # (3.2, sigma=0.7, batch_size/len(data)=100/23574, clipping_bound = 1, iter=10, party=2)

    def train(self, data, epsilon, global_epochs=100,n_critics=10, batch_size=100,
              learning_rate=1e-4, clipping_bound=1):
        """Train the model

        Parameters
        ----------
        data : torch.Tensor
            Data for training
        epochs : int
            Number of iterations over the full data set for training
        n_critics : int
            Number of discriminator training iterations
        batch_size : int
            Number of training examples per inner iteration
        learning_rate : float
            Learning rate for training
        sigma : float or None
            Amount of noise to add (for differential privacy)
        weight_clip : float
            Maximum range of weights (for differential privacy)
        """
        sigma = self.epsilon2sigma[epsilon]
        generator_solver = optim.RMSprop(
            self.generator.parameters(), lr=learning_rate
        )
        discriminator_solver = [optim.RMSprop(
            discriminator.parameters(), lr=learning_rate
        ) for discriminator in self.discriminator]
        
        pic = []
        for epoch in range(global_epochs):
            # logging.info(f'the {epoch}-th epoch is in process')
            dis_loss = 0
            ger_loss = 0
            noise_dim = len(list(self.party2attr[0]))+len(list(self.party2attr[1]))
            noise_for_ger = torch.normal(mean=0,std =1,size=(batch_size, noise_dim))
            ger_gra_dic = dict()
            for param in self.generator.named_parameters():
                ger_gra_dic[param[0]]= torch.zeros_like(param[1])
            for cri in range(n_critics):
                # Sample real data
                rand_perm = torch.randperm(data.size(0))
                samples = data[rand_perm[:batch_size]]
                #generate one noise to syn the fake data
                noise = torch.normal(mean=0,std=1,size=(batch_size, noise_dim))
                for i, discriminator in enumerate(self.discriminator):
                    if epsilon != None:
                        temp_par = dict()
                        for p in discriminator.named_parameters():
                            temp_par[p[0]]= torch.zeros_like(p[1])
                        attr_temp = self.party2attr[i]
                        samples_i = samples[:,attr_temp]
                        # Sample fake data
                        if cuda:
                            real_sample = Variable(samples_i).cuda()
                        else:
                            real_sample = Variable(samples_i)
                        # Score data
                        fake_sample = self.generate_single(noise,i)
                        discriminator_real = discriminator(real_sample)
                        discriminator_fake = discriminator(fake_sample)
                        # Calculate discriminator loss
                        # Discriminator wants to assign a high score to real data
                        # and a low score to fake data
                        discriminator_loss = -discriminator_real+discriminator_fake\
                            +10*self.gradient_penalty(real_sample,fake_sample,discriminator)
                        if cri == n_critics-1:
                            dis_loss += torch.mean(discriminator_loss)
                        for l, loss_l in enumerate(discriminator_loss):
                            discriminator_solver[i].zero_grad()
                            loss_l.backward(retain_graph=l < batch_size - 1)
                            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clipping_bound, norm_type=2)
                            for p in discriminator.named_parameters():
                                temp_par[p[0]]= temp_par[p[0]] + copy.deepcopy(p[1].grad)
                        for p in discriminator.named_parameters():
                            noise_ = torch.normal(mean=0., std=sigma, size=temp_par[p[0]].shape).cuda()
                            p[1].grad = copy.deepcopy(temp_par[p[0]]) + noise_
                            p[1].grad = p[1].grad / batch_size
                    else:
                        attr_temp = self.party2attr[i]
                        samples_i = samples[:,attr_temp]
                        # Sample fake data
                        if cuda:
                            real_sample = Variable(samples_i).cuda()
                        else:
                            real_sample = Variable(samples_i)
                        # Score data
                        fake_sample = self.generate_single(noise,i)
                        discriminator_real = discriminator(real_sample)
                        discriminator_fake = discriminator(fake_sample)
                        # Calculate discriminator loss
                        # Discriminator wants to assign a high score to real data
                        # and a low score to fake data
                        discriminator_loss = -torch.mean(discriminator_real)+torch.mean(discriminator_fake)\
                            +10*self.gradient_penalty(real_sample,fake_sample,discriminator)
                        discriminator_loss.backward()
                        if cri == n_critics-1:
                            dis_loss += discriminator_loss
                    discriminator_solver[i].step()
                    discriminator_solver[i].zero_grad()
                    generator_solver.zero_grad()
                    if  cri == n_critics-1:
                        # Sample and score fake data
                        fake_sample = self.generate_single(noise_for_ger,i)
                        discriminator_fake_ = discriminator(fake_sample)
                        # Calculate generator loss
                        # Generator wants discriminator to assign a high score to fake data
                        generator_loss = -torch.mean(discriminator_fake_)
                        ger_loss += generator_loss
                        generator_loss.backward()
                        for param in self.generator.named_parameters():
                            ger_gra_dic[param[0]]= ger_gra_dic[param[0]] + copy.deepcopy(param[1].grad)
                        pic.append(ger_gra_dic)
                        discriminator_solver[i].zero_grad()
                        generator_solver.zero_grad()
            with torch.no_grad():
                logging.info(f'{epoch+1}-th epoch: discriminator loss:{dis_loss},gererator_loss:{ger_loss}')
            for param in self.generator.named_parameters():
                param[1].grad = copy.deepcopy(ger_gra_dic[param[0]])
            generator_solver.step()
            # Reset gradient
            generator_solver.zero_grad()

    def generate_single(self, noise, i):
        """Generate a synthetic data set using the trained model
        Parameters
        ----------
        n : int
            Number of data points to generate
        Returns
        -------
        torch.Tensor
        """
        if cuda:
            noise = noise.cuda()
        else:
            noise = noise
        fake_sample = self.generator.forward(noise, i)
        return fake_sample
    
    def generate(self, noise):
        """Generate a synthetic data set using the trained model
        Parameters
        ----------
        n : int
            Number of data points to generate
        Returns
        -------
        torch.Tensor
        """
        if cuda:
            noise = noise.cuda()
        else:
            noise = noise
        fake_sample = []
        for i in range(len(self.party2attr)):
            fake_sample.append(self.generator.forward(noise, i))
        fake_sample = torch.hstack(fake_sample)
        return fake_sample.detach().cpu().numpy()
    
    def recover(self,fake_sample):
        output_list = []
        for out in fake_sample:
            out = out.detach().cpu().numpy()
            output_list.append(out)
        recovered_data_numpy = np.hstack((output_list[0],output_list[1]))
        return recovered_data_numpy


    def gradient_penalty(self,fake_data, real_data, discriminator):
        batch_size = real_data.size(0)
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        alpha = Tensor(np.random.random((batch_size,1,1,1)))
        interpolates = ((1 - alpha) * fake_data + alpha * real_data).requires_grad_(True)
        disc_interpolates = discriminator(interpolates)
        if cuda:
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty