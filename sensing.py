import torch

def matched_filtering(signal_grad):
    m, D = signal_grad.shape
    print(m)
    print(D)
    return [0] * m