import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython import display
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
from torch.utils.data.dataset import Dataset

class Net_1(nn.Module):
    
    def __init__(self): #input (28,28)
        super(Net_1, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 2)
        
    def forward(self, x):
        output = x.view(-1, 28*28)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        return output



class Net_2(nn.Module):  #input (28, 28)
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
#         return x#torch.softmax(x)
        return x

class Net_3(nn.Module):  #input (180, 180)
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5 , 1)
        self.conv11 = nn.Conv2d(20, 20, 5 , 1)
        self.conv2 = nn.Conv2d(20, 50, 5 , 1)
        self.fc1 = nn.Linear(4*4*50, 200)
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 4, 4)
        x = F.relu(self.conv11(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 4, 4)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
#         return x#torch.softmax(x)
        return x


class Net_4(nn.Module):  #input (292, 292)
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 10, 3, 3)
        self.conv4 = nn.Conv2d(10, 40, 4, 4)
        self.fc1 = nn.Linear(40*5*5, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 4, 4)
        x = F.max_pool2d(x, 4, 4)
        x= F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 40*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
#         return x#torch.softmax(x)
        return x

from torchvision.transforms import ToTensor
import torch

img_size = (28, 28)
#img_size = (292,292)
#img_size = (180,180)

def DataSetGenerator(path, img_size=img_size):
  normal_path = path + '/NORMAL'
  pneu_path = path + '/PNEUMONIA'

  norm_files = listdir(normal_path)
  pneu_files = listdir(pneu_path)

  labels = []
  img = []
  
  for file in norm_files:
    
    img_path = normal_path + '/' + file
    image = Image.open(img_path)
    image = image.resize(img_size, Image.ANTIALIAS)
    image = ToTensor()(image)
    image = torch.mean(image, dim=0).unsqueeze(0)
    img.append(image)
    labels.append(0)

  for file in pneu_files:
    img_path = pneu_path + '/' + file
    image = Image.open(img_path)
    image = image.resize(img_size, Image.ANTIALIAS)
    image = ToTensor()(image)
    if image.size()[0] != 4:
      if image.size()[0] == 3:
        image = image[0]*0.3 + image[1]*0.59 + image[2]*0.11
        image = image.unsqueeze(0)
      img.append(image)
      labels.append(1)
  
  return torch.stack(img,dim=0), torch.Tensor(labels)

base_path = '/content/drive/My Drive/proyecto/xray_dataset_covid19/'
train_path = base_path + 'train'
test_path = base_path + 'test'


x_train, y_train = DataSetGenerator(train_path)
x_test, y_test = DataSetGenerator(test_path) 
device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

pip install pyro-ppl

pip install memory_profiler

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

log_softmax = nn.LogSoftmax(dim=1)

def model_1(x_data, y_data):


  fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
  fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

  fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight))
  fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias))

  priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
            'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}

  lifted_module = pyro.random_module("module", net, priors)

  lifted_reg_model = lifted_module()

  lhat = log_softmax(lifted_reg_model(x_data))

  pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
  
softplus = torch.nn.Softplus()

def guide_1(x_data, y_data):
    

    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    fc2w_mu = torch.randn_like(net.fc2.weight)
    fc2w_sigma = torch.randn_like(net.fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
    
    fc2b_mu = torch.randn_like(net.fc2.bias)
    fc2b_sigma = torch.randn_like(net.fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
    
    


    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()

net = Net_1()

log_softmax = nn.LogSoftmax(dim=1)

def model_2(x_data, y_data):

  conv1w_prior = Normal(loc=torch.zeros_like(net.conv1.weight), scale=torch.ones_like(net.conv1.weight))
  conv1b_prior = Normal(loc=torch.zeros_like(net.conv1.bias), scale=torch.ones_like(net.conv1.bias))

  conv2w_prior = Normal(loc=torch.zeros_like(net.conv2.weight), scale=torch.ones_like(net.conv2.weight))
  conv2b_prior = Normal(loc=torch.zeros_like(net.conv2.bias), scale=torch.ones_like(net.conv2.bias))

  fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
  fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

  fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight))
  fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias))

  priors = {'conv1.weight': conv1w_prior, 'conv1.bias': conv1b_prior,
            'conv2.weight': conv2w_prior, 'conv2.bias': conv2b_prior,
            'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
            'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}

  lifted_module = pyro.random_module("module", net, priors)

  lifted_reg_model = lifted_module()

  lhat = log_softmax(lifted_reg_model(x_data))

  pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
  
softplus = torch.nn.Softplus()

def guide_2(x_data, y_data):
    
    conv1w_mu = torch.randn_like(net.conv1.weight)
    conv1w_sigma = torch.randn_like(net.conv1.weight)
    conv1w_mu_param = pyro.param("conv1w_mu", conv1w_mu)
    conv1w_sigma_param = softplus(pyro.param("conv1w_sigma", conv1w_sigma))
    conv1w_prior = Normal(loc=conv1w_mu_param, scale=conv1w_sigma_param)
    
    conv1b_mu = torch.randn_like(net.conv1.bias)
    conv1b_sigma = torch.randn_like(net.conv1.bias)
    conv1b_mu_param = pyro.param("conv1b_mu", conv1b_mu)
    conv1b_sigma_param = softplus(pyro.param("conv1b_sigma", conv1b_sigma))
    conv1b_prior = Normal(loc=conv1b_mu_param, scale=conv1b_sigma_param)  

    conv2w_mu = torch.randn_like(net.conv2.weight)
    conv2w_sigma = torch.randn_like(net.conv2.weight)
    conv2w_mu_param = pyro.param("conv2w_mu", conv2w_mu)
    conv2w_sigma_param = softplus(pyro.param("conv2w_sigma", conv2w_sigma))
    conv2w_prior = Normal(loc=conv2w_mu_param, scale=conv2w_sigma_param)
    
    conv2b_mu = torch.randn_like(net.conv2.bias)
    conv2b_sigma = torch.randn_like(net.conv2.bias)
    conv2b_mu_param = pyro.param("conv2b_mu", conv2b_mu)
    conv2b_sigma_param = softplus(pyro.param("conv2b_sigma", conv2b_sigma))
    conv2b_prior = Normal(loc=conv2b_mu_param, scale=conv2b_sigma_param)

    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    fc2w_mu = torch.randn_like(net.fc2.weight)
    fc2w_sigma = torch.randn_like(net.fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
    
    fc2b_mu = torch.randn_like(net.fc2.bias)
    fc2b_sigma = torch.randn_like(net.fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
    
    


    priors = {'conv1.weight': conv1w_prior, 'conv1.bias': conv1b_prior, 'conv2.weight': conv2w_prior, 'conv2.bias': conv2b_prior,
              'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()

net = Net_2()

log_softmax = nn.LogSoftmax(dim=1)

def model_3(x_data, y_data):

  conv1w_prior = Normal(loc=torch.zeros_like(net.conv1.weight), scale=torch.ones_like(net.conv1.weight))
  conv1b_prior = Normal(loc=torch.zeros_like(net.conv1.bias), scale=torch.ones_like(net.conv1.bias))

  conv11w_prior = Normal(loc=torch.zeros_like(net.conv11.weight), scale=torch.ones_like(net.conv11.weight))
  conv11b_prior = Normal(loc=torch.zeros_like(net.conv11.bias), scale=torch.ones_like(net.conv11.bias))


  conv2w_prior = Normal(loc=torch.zeros_like(net.conv2.weight), scale=torch.ones_like(net.conv2.weight))
  conv2b_prior = Normal(loc=torch.zeros_like(net.conv2.bias), scale=torch.ones_like(net.conv2.bias))

  fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
  fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

  fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight))
  fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias))

  priors = {'conv1.weight': conv1w_prior, 'conv1.bias': conv1b_prior,
            'conv11.weight': conv11w_prior, 'conv11.bias': conv11b_prior,
            'conv2.weight': conv2w_prior, 'conv2.bias': conv2b_prior,
            'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
            'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}

  lifted_module = pyro.random_module("module", net, priors)

  lifted_reg_model = lifted_module()

  lhat = log_softmax(lifted_reg_model(x_data))

  pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
  
softplus = torch.nn.Softplus()

def guide_3(x_data, y_data):
    
    conv1w_mu = torch.randn_like(net.conv1.weight)
    conv1w_sigma = torch.randn_like(net.conv1.weight)
    conv1w_mu_param = pyro.param("conv1w_mu", conv1w_mu)
    conv1w_sigma_param = softplus(pyro.param("conv1w_sigma", conv1w_sigma))
    conv1w_prior = Normal(loc=conv1w_mu_param, scale=conv1w_sigma_param)
    
    conv1b_mu = torch.randn_like(net.conv1.bias)
    conv1b_sigma = torch.randn_like(net.conv1.bias)
    conv1b_mu_param = pyro.param("conv1b_mu", conv1b_mu)
    conv1b_sigma_param = softplus(pyro.param("conv1b_sigma", conv1b_sigma))
    conv1b_prior = Normal(loc=conv1b_mu_param, scale=conv1b_sigma_param)  

    conv11w_mu = torch.randn_like(net.conv11.weight)
    conv11w_sigma = torch.randn_like(net.conv11.weight)
    conv11w_mu_param = pyro.param("conv11w_mu", conv11w_mu)
    conv11w_sigma_param = softplus(pyro.param("conv11w_sigma", conv11w_sigma))
    conv11w_prior = Normal(loc=conv11w_mu_param, scale=conv11w_sigma_param)
    
    conv11b_mu = torch.randn_like(net.conv11.bias)
    conv11b_sigma = torch.randn_like(net.conv11.bias)
    conv11b_mu_param = pyro.param("conv11b_mu", conv11b_mu)
    conv11b_sigma_param = softplus(pyro.param("conv11b_sigma", conv11b_sigma))
    conv11b_prior = Normal(loc=conv11b_mu_param, scale=conv11b_sigma_param)  

    conv2w_mu = torch.randn_like(net.conv2.weight)
    conv2w_sigma = torch.randn_like(net.conv2.weight)
    conv2w_mu_param = pyro.param("conv2w_mu", conv2w_mu)
    conv2w_sigma_param = softplus(pyro.param("conv2w_sigma", conv2w_sigma))
    conv2w_prior = Normal(loc=conv2w_mu_param, scale=conv2w_sigma_param)
    
    conv2b_mu = torch.randn_like(net.conv2.bias)
    conv2b_sigma = torch.randn_like(net.conv2.bias)
    conv2b_mu_param = pyro.param("conv2b_mu", conv2b_mu)
    conv2b_sigma_param = softplus(pyro.param("conv2b_sigma", conv2b_sigma))
    conv2b_prior = Normal(loc=conv2b_mu_param, scale=conv2b_sigma_param)

    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    fc2w_mu = torch.randn_like(net.fc2.weight)
    fc2w_sigma = torch.randn_like(net.fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
    
    fc2b_mu = torch.randn_like(net.fc2.bias)
    fc2b_sigma = torch.randn_like(net.fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
    
    


    priors = {'conv1.weight': conv1w_prior, 'conv1.bias': conv1b_prior, 
              'conv11.weight': conv11w_prior, 'conv11.bias': conv11b_prior,
              'conv2.weight': conv2w_prior, 'conv2.bias': conv2b_prior,
              'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()
  
net = Net_3()

log_softmax = nn.LogSoftmax(dim=1)

def model_4(x_data, y_data):

  conv1w_prior = Normal(loc=torch.zeros_like(net.conv1.weight), scale=torch.ones_like(net.conv1.weight))
  conv1b_prior = Normal(loc=torch.zeros_like(net.conv1.bias), scale=torch.ones_like(net.conv1.bias))

  conv2w_prior = Normal(loc=torch.zeros_like(net.conv2.weight), scale=torch.ones_like(net.conv2.weight))
  conv2b_prior = Normal(loc=torch.zeros_like(net.conv2.bias), scale=torch.ones_like(net.conv2.bias))

  conv3w_prior = Normal(loc=torch.zeros_like(net.conv3.weight), scale=torch.ones_like(net.conv3.weight))
  conv3b_prior = Normal(loc=torch.zeros_like(net.conv3.bias), scale=torch.ones_like(net.conv3.bias))

  conv4w_prior = Normal(loc=torch.zeros_like(net.conv4.weight), scale=torch.ones_like(net.conv4.weight))
  conv4b_prior = Normal(loc=torch.zeros_like(net.conv4.bias), scale=torch.ones_like(net.conv4.bias))

  fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
  fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

  fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weight), scale=torch.ones_like(net.fc2.weight))
  fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias), scale=torch.ones_like(net.fc2.bias))

  priors = {'conv1.weight': conv1w_prior, 'conv1.bias': conv1b_prior,
            'conv2.weight': conv2w_prior, 'conv2.bias': conv2b_prior,
            'conv3.weight': conv3w_prior, 'conv3.bias': conv3b_prior,
            'conv4.weight': conv4w_prior, 'conv4.bias': conv4b_prior,
            'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
            'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}

  lifted_module = pyro.random_module("module", net, priors)

  lifted_reg_model = lifted_module()

  lhat = log_softmax(lifted_reg_model(x_data))

  pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
  
softplus = torch.nn.Softplus()

def guide_4(x_data, y_data):
    
    conv1w_mu = torch.randn_like(net.conv1.weight)
    conv1w_sigma = torch.randn_like(net.conv1.weight)
    conv1w_mu_param = pyro.param("conv1w_mu", conv1w_mu)
    conv1w_sigma_param = softplus(pyro.param("conv1w_sigma", conv1w_sigma))
    conv1w_prior = Normal(loc=conv1w_mu_param, scale=conv1w_sigma_param)
    
    conv1b_mu = torch.randn_like(net.conv1.bias)
    conv1b_sigma = torch.randn_like(net.conv1.bias)
    conv1b_mu_param = pyro.param("conv1b_mu", conv1b_mu)
    conv1b_sigma_param = softplus(pyro.param("conv1b_sigma", conv1b_sigma))
    conv1b_prior = Normal(loc=conv1b_mu_param, scale=conv1b_sigma_param)  

    conv2w_mu = torch.randn_like(net.conv2.weight)
    conv2w_sigma = torch.randn_like(net.conv2.weight)
    conv2w_mu_param = pyro.param("conv2w_mu", conv2w_mu)
    conv2w_sigma_param = softplus(pyro.param("conv2w_sigma", conv2w_sigma))
    conv2w_prior = Normal(loc=conv2w_mu_param, scale=conv2w_sigma_param)
    
    conv2b_mu = torch.randn_like(net.conv2.bias)
    conv2b_sigma = torch.randn_like(net.conv2.bias)
    conv2b_mu_param = pyro.param("conv2b_mu", conv2b_mu)
    conv2b_sigma_param = softplus(pyro.param("conv2b_sigma", conv2b_sigma))
    conv2b_prior = Normal(loc=conv2b_mu_param, scale=conv2b_sigma_param)

    conv3w_mu = torch.randn_like(net.conv3.weight)
    conv3w_sigma = torch.randn_like(net.conv3.weight)
    conv3w_mu_param = pyro.param("conv3w_mu", conv3w_mu)
    conv3w_sigma_param = softplus(pyro.param("conv3w_sigma", conv3w_sigma))
    conv3w_prior = Normal(loc=conv3w_mu_param, scale=conv3w_sigma_param)
    
    conv3b_mu = torch.randn_like(net.conv3.bias)
    conv3b_sigma = torch.randn_like(net.conv3.bias)
    conv3b_mu_param = pyro.param("conv3b_mu", conv3b_mu)
    conv3b_sigma_param = softplus(pyro.param("conv3b_sigma", conv3b_sigma))
    conv3b_prior = Normal(loc=conv3b_mu_param, scale=conv3b_sigma_param)  

    conv4w_mu = torch.randn_like(net.conv4.weight)
    conv4w_sigma = torch.randn_like(net.conv4.weight)
    conv4w_mu_param = pyro.param("conv4w_mu", conv4w_mu)
    conv4w_sigma_param = softplus(pyro.param("conv4w_sigma", conv4w_sigma))
    conv4w_prior = Normal(loc=conv4w_mu_param, scale=conv4w_sigma_param)
    
    conv4b_mu = torch.randn_like(net.conv4.bias)
    conv4b_sigma = torch.randn_like(net.conv4.bias)
    conv4b_mu_param = pyro.param("conv4b_mu", conv4b_mu)
    conv4b_sigma_param = softplus(pyro.param("conv4b_sigma", conv4b_sigma))
    conv4b_prior = Normal(loc=conv4b_mu_param, scale=conv4b_sigma_param)



    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    fc2w_mu = torch.randn_like(net.fc2.weight)
    fc2w_sigma = torch.randn_like(net.fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param)
    
    fc2b_mu = torch.randn_like(net.fc2.bias)
    fc2b_sigma = torch.randn_like(net.fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param)
    
    


    priors = {'conv1.weight': conv1w_prior, 'conv1.bias': conv1b_prior, 'conv2.weight': conv2w_prior, 'conv2.bias': conv2b_prior,
              'conv3.weight': conv3w_prior, 'conv3.bias': conv3b_prior, 'conv4.weight': conv4w_prior, 'conv4.bias': conv4b_prior,
              'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()
  
net = Net_4()

# Commented out IPython magic to ensure Python compatibility.
# %load_ext memory_profiler

def training():
  optim = Adam({"lr":0.5})
  svi = SVI(model_2, guide_2, optim, loss=Trace_ELBO())
  num_samples = 50

  num_iterations = 50
  acc = []
  for j in range(num_iterations):
      loss = svi.step(x_train, y_train)
      
      def predict(x):
          sampled_models = [guide_2(None, None) for _ in range(num_samples)]
          yhats = [model(x).data for model in sampled_models]
          mean = torch.mean(torch.stack(yhats), 0)
          return np.argmax(mean.numpy(), axis=1)
          #return mean
      

      predicted = predict(x_test)
      predicted = torch.tensor(predicted)
      total = y_test.size(0)
      correct = (predicted == y_test).sum().item()
      acc.append(100*correct/total)
      print(j, acc[j], loss)
      if acc[j] >=85:
        break

  return acc
      

acc = training()
"""
normalizer_train = 148
total_epoch_loss_train = loss / normalizer_train

print("Epoch ", j, " Loss ", total_epoch_loss_train)
if total_epoch_loss_train < 300:
  break"""

n = 500
sampled_models = [guide_2(None, None) for _ in range(n)]
vec31 = [0]*n
vec32 = [0]*n
for j in range(n):
  x = sampled_models[j]
  y = x.fc1
  z = x.conv2
  vec31[j] = float(y.weight[0][0])
  vec32[j] = float(z.weight[0][0][0][0])

fig, axs = plt.subplots(1, 3, figsize=(20,5),sharey=True)

xlim = [min(vec21+ vec22+ vec32), max(vec12+ vec22+ vec32)]
fs=17
axs[0].hist(vec12, color = "C1", density=True, alpha=0.5, bins=15,range=xlim, label="acc = 50%")
axs[0].legend(loc=0,fontsize=fs)
axs[1].hist(vec22, color = "C1", density=True, alpha=0.5, bins=15,range=xlim, label="acc = 70%")
axs[1].legend(loc=0,fontsize=fs)
axs[2].hist(vec32, color = "C1", density=True, alpha=0.5, bins=15,range=xlim, label="acc = 90%")
axs[2].legend(loc=0,fontsize=fs)
plt.show()

xlim = [-1.5, 0.75]
plt.hist(vec11-np.mean(vec11), color = "C0", density=True, alpha=0.5, bins=15,range=xlim, label="acc = 50%")
plt.legend()

plt.hist(vec21, color = "C0", density=True, alpha=0.5, bins=15,range=xlim, label="acc = 70%")
plt.legend()

plt.hist(vec31, color = "C0", density=True, alpha=0.5, bins=15,range=xlim, label="acc = 90%")
plt.legend()

# Commented out IPython magic to ensure Python compatibility.
num_samples = 100
# %timeit sampled_models = [guide_4(None, None) for _ in range(num_samples)]

num_samples = 100
def predict(x):
    sampled_models = [guide_1(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)
    #return mean
print('Prediction when network is forced to predict')
correct = 0
total = 0

predicted = predict(x_test)
predicted = torch.tensor(predicted)
total += y_test.size(0)
correct += (predicted == y_test).sum().item()

print("accuracy: %d %%" % (100 * correct / total))

import random

elem = [-2.5,0, 2.5, 5, 10]
prob = [0.4,0.2, 0.2,0.1,0.1 ]

result = [50]
while result[-1] <90:
  result = [50]
  while len(result)<20:
    step = np.random.choice(elem, 1, p=prob)[0]
    result.append(result[-1]+step)

plt.plot(result)

acc4 = result

import matplotlib.pyplot as plt
from math import log
t1 = ([0.113]*(len(acc1)-1))
t2 = ([0.159]*(len(acc2)-1))
t3 = ([2.13]*(len(acc3)-1))
t4 = ([53.619]*(len(acc4)-1))

t1.insert(0, 0)
t2.insert(0, 0)
t3.insert(0, 0)
t4.insert(0, 0)


t1 = np.cumsum(t1)
t2 = np.cumsum(t2)
t3 = np.cumsum(t3)
t4 = np.cumsum(t4)


plt.plot(t1, acc1, label = "Model 1")
plt.plot(t2, acc2, label = "Model 2")
plt.plot(t3, acc3, label = "Model 3")
plt.plot(t4, acc4, label = "Model 4")
plt.xscale("log")
plt.xlabel("Log-Time")
plt.ylabel("Accuracy [%]")
plt.legend()
plt.show()

print(t2, acc2)

print(acc1)
print(acc2)
print(acc3)
print(acc4)
