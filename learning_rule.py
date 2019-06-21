# Update rule RLS
import torch


def RLS(loss_step, P, r):

    """ calculates the update matrix P and new 'gradients' for the synapses and weight matrices
        according to Recursive Least Squares algorithm

    Arguments:
        loss(required, float): error between the current output and target signal
        P(required, tensor): matrix for update, dimension(number of neurons, number of neurons)
        r(required, tensor): firing rates of neurons on previous step, dimension(number of neurons, 1)
    Outputs:
        P (tensor, (number of neurons, number of neurons)): matrix for update (for the next time step)
        dweight (tensor, (number of neurons, 1)): update for the weight and synapses matrices

    """
    k = torch.matmul(P, r)
    rPr = torch.matmul(r.view(1, -1), k)
    c = 1.0 / (1.0 + rPr)
    P = P - torch.matmul(k, (k.view(1, -1) * c))
    dweight = -loss_step * k * c
    return P, dweight

