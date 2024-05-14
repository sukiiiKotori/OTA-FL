import numpy as np

def optimal_power():
    p_u:float
    p_v:float
    p_g:float
    return 1, 1, 1

def optimal_power_selection():
    p_u:float
    p_v:float
    p_g:float
    M:list
    return p_u, p_v, p_g, M

def optimal_beamforming():
    #to be continue
    return []

def get_args(var_receive, data_per_user, P_g, F, H):  
    inner = np.array(F).conj() @ H #(1xN @ NxM = 1xM)
    eta_arr = (np.abs(inner) / (np.array(data_per_user) * var_receive))**2
    #get eta
    eta = P_g * np.min(eta_arr)
    #get p_gm
    p_gm = np.array(data_per_user) * np.sqrt(eta) * var_receive / inner
    return p_gm, eta