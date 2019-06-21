import torch
import numpy as np
import random


def gen_delay_trial(lengthStimulus, delay, lengthSignal, pattern='random', exp_speed=40):

    """"generates 2 input signals for the network and target signal for 'match-to-sample' task

    Arguments:
        lengthStimulus (required, int): length of a/b stimulus, A stimuli [1,0], B stimuli [0,1]
        delay(required, int): delay between stimulus
        lengthSignal(required, int): length of the trial
        pattern (optional, str): defines the order of stimulus and the type of trial respectively:
            'random' - random order, 'AA', 'AB', 'BB', 'BA'. Default 'random'.
        exp_speed (optional, int): defines the steepness of the target function. Default 40.


    Output:
        u1 (list): signal for the first input
        u2 (list): signal for the second input
        target (list): target signal
    """

    stimulus = [[1, 0], [0, 1]]
    st1 = random.choice(stimulus)
    st2 = random.choice(stimulus)


    stim_lib = {'random': [st1, st2], 'AA': [stimulus[0], stimulus[0]], 'AB': [stimulus[0], stimulus[1]],
                'BB': [stimulus[1], stimulus[1]], 'BA': [stimulus[1], stimulus[0]]}

    st1 = stim_lib[pattern][0]
    st2 = stim_lib[pattern][1]

    u1 = [st1[0] for k in range(lengthStimulus)] + [0 for k in range(delay)] + [st2[0] for k in range(lengthStimulus)] +\
         [0 for k in range(lengthSignal - 2 * lengthStimulus - delay)]
    u2 = [st1[1] for k in range(lengthStimulus)] + [0 for k in range(delay)] + [st2[1] for k in range(lengthStimulus)] + \
         [0 for k in range(lengthSignal - 2 * lengthStimulus - delay)]
    start_target = delay + lengthStimulus * 2
    if st1 == st2:
        target = [0 for k in range(lengthStimulus * 2 + delay)] + [-np.exp(-(n - start_target) / exp_speed) + 1 for n in
                                                                   range(start_target, lengthSignal)]

    else:
        target = [0 for k in range(lengthStimulus * 2 + delay)] + [np.exp(-(n - start_target) / exp_speed) - 1 for n in
                                                                   range(start_target, lengthSignal)]

    return u1, u2, target


def match_to_sample_input(num_trials, lengthStimulus, delay, lengthSignal, pattern='random', exp_speed=40):
    """generates 2 input signals for the network and target signal for 'match-to-sample' task
        for a given number of trials

    Arguments:
        num_trials (required, int): number of trials to generate
        lengthStimulus (required, int): length of a/b stimulus
        delay (required, int) - delay between stimulus
        lengthSignal(required, int): length of the trial
        pattern (optional, str): defines the order of stimulus and the type of trial respectively:
            'random' - random order, 'AA', 'AB', 'BB', 'BA'. Default 'random'.
        exp_speed (optional, int): defines the steepness of the target function. Default 40.

    Output:
        signal(torch tensor, (2,num_trials*lengthSignal)): input signal with num_trials trials
        target(torch tensor, (1,num_trials*lengthSignal)): target signal for the num_trials trials
        trial_target (list, (1, num_trials)): labels for trials' output 0 or 1.
    """
    signal_u1 = np.array([])
    signal_u2 = np.array([])
    target = np.array([])
    trial_target = np.empty(num_trials)
    for tr in range(num_trials):
        u1, u2, target_tr = gen_delay_trial(lengthStimulus, delay, lengthSignal, pattern)
        signal_u1 = np.append(signal_u1, u1)
        signal_u2 = np.append(signal_u2, u2)
        target = np.append(target, target_tr)
        trial_target[tr] = np.sign(target[-1])
    signal = np.array([signal_u1, signal_u2])
    # target = torch.Tensor(target)

    return signal, target, trial_target


def repeat_sequence_input(num_trials):
    """generates input for the network and target signal for repeat-sequence task

    'Music notes' for 4 strings: EGDC, according to the method proposed in
    'Supervised learning in spiking neural networks with FORCE training'
    W.Nicola and C.Clopath, Nature 2017, https://www.nature.com/articles/s41467-017-01827-3

    Arguments:
        num_trials(required, int) - number of trials

    Outputs:
        signal(torch tensor, (4, 2800 * num_trials)) - input signal (for 4 inputs)
        target(torch tensor, (4, 2800 * num_trials)) - target signal (for 4 outputs)
    """

    half = [np.sin(2 * np.pi * t / 200) for t in range(100)]
    whole = [np.sin(2 * np.pi * t / 400) for t in range(200)]
    pause = [0 for k in range(100)]
    response_time = np.zeros([4, 1400])
    signal = np.empty([4, 2800 * num_trials])
    target = np.empty([4, 2800 * num_trials])
    for tr in range(num_trials):
        signal_tr = np.empty([4, 1400])
        E = np.concatenate((half, half, whole, pause, half, half, whole, pause, half, pause, pause, pause))
        G = np.concatenate(
            (pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, half, pause, pause))
        D = np.concatenate(
            (pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, half))
        C = np.concatenate(
            (pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, pause, half, pause))
        signal_tr[0, :] = G
        signal_tr[1, :] = E
        signal_tr[2, :] = D
        signal_tr[3, :] = C
        target_tr = np.concatenate((response_time, signal_tr), axis=1)
        signal_tr = np.concatenate((signal_tr, response_time), axis=1)
        signal[:, tr * 2800:(tr + 1) * 2800] = signal_tr
        target[:, tr * 2800:(tr + 1) * 2800] = target_tr

        return torch.Tensor(signal), torch.Tensor(target)
