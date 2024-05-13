import numpy as np

def transmission_var(var_locals, mean_locals, P_v, F, H, v_max, noise_v):
    noise_factor = np.array(F).conj() @ noise_v #(1xN @ NxM = 1xM)
    inner = np.array(F).conj() @ H #(1xN @ NxM = 1xM)
    p_v = np.sqrt(P_v)
    # formula (17)
    receive_signals = np.array(var_locals)*np.sign(np.array(mean_locals)) + (v_max * noise_factor) / (p_v * inner)
    return np.abs(receive_signals), np.sign(np.real(receive_signals))

def transmission_mean(mean_locals, dist_locals, P_u, F, H, u_max, dist_max, noise_u):
    noise_factor = np.array(F).conj() @ noise_u #(1xN @ NxM = 1xM)
    inner = np.array(F).conj() @ H #(1xN @ NxM = 1xM)
    p_u = np.sqrt(P_u)
    theta = np.array(dist_locals) / dist_max * 2*np.pi
    # formula (16)
    receive_signals = np.abs(np.array(mean_locals)) * np.exp(1j*theta) + (u_max * noise_factor) / (p_u * inner)
    return np.abs(receive_signals), np.angle(receive_signals) / 2*np.pi * dist_max

def transmit_grad(grads, args):
    pass

def receive_grad():
    pass