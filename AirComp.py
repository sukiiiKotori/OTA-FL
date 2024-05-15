import numpy as np
from collections import OrderedDict

def transmission_var(var_locals, mean_locals, P_v, F, H, v_max, sigma_n):
    N, M = H.shape
    noise = (np.random.randn(N, M)+1j*np.random.randn(N, M)) / 2**0.5 * sigma_n
    noise_factor = np.array(F).conj() @ noise #(1xN @ NxM = 1xM)
    inner = np.array(F).conj() @ H #(1xN @ NxM = 1xM)
    p_v = np.sqrt(P_v)
    # formula (17)
    receive_signals = var_locals * np.sign(mean_locals) + (v_max * noise_factor) / (p_v * inner)
    return np.abs(receive_signals), np.sign(np.real(receive_signals))

def transmission_mean(mean_locals, dist_locals, P_u, F, H, u_max, dist_max, sigma_n):
    N, M = H.shape
    noise = (np.random.randn(N, M)+1j*np.random.randn(N, M)) / 2**0.5 * sigma_n
    noise_factor = np.array(F).conj() @ noise #(1xN @ NxM = 1xM)
    inner = np.array(F).conj() @ H #(1xN @ NxM = 1xM)
    p_u = np.sqrt(P_u)
    theta = np.array(dist_locals) / dist_max * 2*np.pi
    # formula (16)
    receive_signals = np.abs(np.array(mean_locals)) * np.exp(1j*theta) + (u_max * noise_factor) / (p_u * inner)
    return np.abs(receive_signals), np.angle(receive_signals) / 2*np.pi * dist_max

def transmit_grad(grad_locals, mean_locals, var_locals, p_gm):
    # formula (9)
    normal_grad =  (grad_locals - np.array(mean_locals).reshape(-1,1)) / np.array(var_locals).reshape(-1,1)
    return np.array(p_gm).reshape(-1,1) * normal_grad

def receive_grad(F, H, signal_grad, eta, sigma_n):
    N, M = H.shape
    _, D = signal_grad.shape
    noise = (np.random.randn(N,D)+1j*np.random.randn(N,D)) / 2**0.5 * sigma_n
    # formula (13)
    signals = H @ signal_grad + noise # nxD = nxm @ mxD
    receive_signals = np.array(F).conj() @ signals # 1xD = 1xn @ nxD
    return np.real(receive_signals) / eta**0.5
        