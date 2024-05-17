import numpy as np
import cvxpy as cp

def optimal_power(P_max, beta_1, beta_2, data_per_user, F, H, u_max, v_max, D, sigma_n):
    # not condsider user selection
    user_list = [i for i in range(len(data_per_user))]
    """ sum_data = sum(data_per_user)
    inner = np.array(F).conj() @ H #(1xN @ NxM = 1xM)
    term_1 = D * sigma_n**2 / sum_data**2
    term_2 = (np.array(data_per_user) / np.abs(inner))**2
    a = term_1 * u_max**2 * np.sum(term_2)
    b = term_1 * v_max**2 * np.sum(term_2)
    c = term_1 * v_max**2 * (D * sigma_n**2) * np.max(term_2 / np.abs(inner)**2)
    d = term_1 * np.max(term_2)
    print(sigma_n**2)
    print(a,b,c,d)
    rho = 0
    P_G = cp.Variable()
    objective = cp.Minimize(((a+b+c/P_G+2*(a*b+a*c/P_G)**0.5)/(P_max-P_G)+beta_1/(2*beta_2)) / (1-2*beta_2*(rho+d/P_G)))
    constraints = [
        (1 - 2 * beta_2 * rho) / (2 * beta_2 * d) <= P_G,
        P_G <= P_max 
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    P_G = P_G.value
    P_u = (P_max - P_G)*((a*P_G)**0.5/((a*P_G)**0.5 + (b*P_G+c)**0.5))
    P_v = P_max - P_G - P_u """

    return 200, 1000, 1000, user_list

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