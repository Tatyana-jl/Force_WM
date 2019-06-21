import torch
import numpy as np
from torch.autograd import Variable
from RNN_neurons import RnnForNeurons
from learning_rule import RLS


# class RunModFORCE:
#
#     # class RunMode performs the 'experiment'. It creates the network with assigned parameters:
#
#     # num_neurons: number of neurons in the 'pool'
#     # output_size: number of the output channels
#     # num_inputs: number of the input channels
#     # tau: relexation constant for neurons
#     # g: synaptic scaling
#
#     def __init__(self, num_neurons, output_size, num_inputs, tau, g):
#         self.num_neurons = num_neurons
#         self.tau = tau
#         self.g = g
#         self.output_size = output_size
#         self.net = RnnForNeurons(num_neurons=self.num_neurons, output_size=self.output_size, tau=self.tau, g=self.g)
#
#         # # input weights - created once for the network and stay fixed
#         # self.weightInput = torch.distributions.normal.Normal(0, 1).sample((self.num_neurons, num_inputs))

def run(network, input_net, return_measures=False, train=False, delta_t=2, alpha=1, lengthSignal=1000, target=None,
        check_error_update=False):
    """ runs the network based on the input signal and parameters of training/testing,
        learning rule: RLS

    Arguments:
        network(required): network model
        input_net(required, numpy array): input signal, with dimension (number of inputs, length of input)
        return_measures(optional, bool): if True - record potentials during run, default - False
        train(optional, bool): if True - perform the synapses and weights update, default - False
        delta_t(optional, int): time interval for synapses and weights update in training mode, default - 2ms
        alpha(optional, float): learning rate, default - 1
        lengthSignal(optional, int):  defines the time of potentials dropping (corresponds to the new trial),
                                    default - 1000ms
        target(optional, tensor): target signal for learning mode, default - None
        check_error_update(optional, bool): check the error on the same input with updated weights if True, default-False

    Output:
        output(tensor, (number of outputs, lengthSignal*number_if_trials)): output signal from the network
        potentials_record (tensor, (number of neurons, lengthSignal*number_if_trials)): potentials of the neurons in
                        the 'pool' during run
    """
    cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    if cuda:
        network.cuda()
        if target is not None:
            target = Variable(FloatTensor(target), requires_grad=False)

    length_input = input_net.shape[1]
    output = FloatTensor(length_input)

    # For the loss:
    loss = np.array([])
    # measure output potentials
    potentials_record = torch.Tensor(network.num_neurons, length_input)

    if train:
        # Create matrices P_w for synapses update
        p_w = FloatTensor(np.eye(network.num_neurons) / alpha)

    input_net = Variable(FloatTensor(input_net), requires_grad=False)

    with torch.no_grad():
        for t in range(length_input):
            if t % lengthSignal == 0:
                # initialize potentials with uniform noise in the [-0.1,0.1] at each trial (according to the paper)
                potentials = Variable(torch.distributions.normal.Normal(0, 0.5).sample((network.num_neurons, 1)).cuda())

            output[t], potentials = network(input_net[:, t], potentials)
            r_post = torch.tanh(potentials)

            if return_measures:
                potentials_record[:, t] = potentials.view(1, -1)

            # update the weights on each delta_t
            if train & (t % delta_t == 0):
                loss_trial = output[t] - target[t]
                loss = np.append(loss, loss_trial)
                # Calculate update
                p_w, dW = RLS(loss_trial, p_w, r_post)
                # Update parameters
                network.state_dict()['gen_layer.synapses'].data.add_(dW.float())
                network.state_dict()['read_out.weight'].data.add_(dW.float())
    if return_measures:
        return output, potentials_record
    else:
        return output
