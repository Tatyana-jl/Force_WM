import torch
import torch.nn as nn
from custom_layers import Neurons, LinearModified
import numpy as np


class RnnForNeurons(nn.Module):
    """class for Neural Network with two layers: RNN pool and read-out

    Arguments:
         num_neurons(required, int): number of neurons in the 'pool'
         input_size(required, int): number of inputs
         output_size(required, int): number of output channels
         tau(required, float): relaxation constant for neurons in the 'pool'
         g(required, float): synaptic scaling for neurons in the 'pool'
    """

    def __init__(self, num_neurons, input_size, output_size, tau, g, p=1):
        super(RnnForNeurons, self).__init__()
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.output_size = output_size
        self.tau = tau
        self.g = g
        self.p = p

        # Neurons and LinearModified - are custom layers (see the next section)
        self.input_map = LinearModified(num_neurons=self.input_size, output_size=self.num_neurons)
        self.gen_layer = Neurons(num_neurons=self.num_neurons, tau=self.tau, g=self.g)
        self.read_out = LinearModified(num_neurons=self.num_neurons, output_size=self.output_size)

        # Initialization
        # input weights - created once for the network and stay fixed
        torch.nn.init.normal_(self.input_map.weight, 0, 0.1)
        # self.input_map.weight.requires_grad = False

        # self.input_map.weight = torch.distributions.normal.Normal(0, 1).sample((self.num_neurons, self.input_size))

        # synapses matrix initialized with weights taken from a normal distribution with mean 0 and var 1/sqrt(N)
        torch.nn.init.normal_(self.gen_layer.synapses, 0, 1 / np.sqrt(self.num_neurons))

        # W initialized either to zero or to values generated by a Gaussian distribution
        # with zero mean and variance (1/(pN))
        torch.nn.init.normal_(self.read_out.weight, 0, 1 / num_neurons)
        # torch.nn.init.zeros_(self.read_out.weight)

    def forward(self, inputs, potentials):
        inputs_mapped = self.input_map(inputs)
        inputs_mapped = torch.unsqueeze(inputs_mapped, dim=0)
        genNet, potentials = self.gen_layer(inputs_mapped, potentials)
        output = self.read_out(genNet)
        return output, potentials
