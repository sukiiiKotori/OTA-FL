import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
#from main_fed import *
#from options import *
#from sampling import *
from datetime import datetime

def Golden_Search(f, l, r, tol=1e-10, max_iter=10000):
    a = l+tol
    b = r-tol

    def calculate_lambda(a, b, ratio=0.382):
        return a + ratio * (b - a)

    def calculate_mu(a, b, ratio=0.618):
        return a + ratio * (b - a)

    Mui = calculate_mu(a, b)
    Mui_value = f(Mui)

    Lamb = calculate_lambda(a, b)
    Lamb_value= f(Lamb)


    for _ in range(max_iter): #判断是否到达终止线
        if abs(f(a)-f(b)) < tol:
            break
        elif Lamb_value < Mui_value:
            b = Mui
            Mui = Lamb
            Lamb = calculate_lambda(a, b)
        else:
            a = Lamb
            Lamb = Mui
            Mui = calculate_mu(a, b)

        Mui_value = f(Mui)
        Lamb_value = f(Lamb)

    return (a+b)/2

def optimal_power(P_max, beta_1, beta_2, data_per_user, F, H, u_max, v_max, D, sigma_n):
    # not condsider user selection
    user_list = [i for i in range(len(data_per_user))]
    sum_data = sum(data_per_user)
    inner = np.array(F).conj() @ H #(1xN @ NxM = 1xM)
    term_1 = D * sigma_n**2 / sum_data**2
    term_2 = (np.array(data_per_user) / np.abs(inner))**2
    a = term_1 * u_max**2 * np.sum(term_2)
    b = term_1 * v_max**2 * np.sum(term_2)
    c = term_1 * v_max**2 * (D * sigma_n**2) * np.max(term_2 / np.abs(inner)**2)
    d = term_1 * np.max(term_2)
    rho = 0
    #print(sigma_n**2)
    print("a,b,c,d,rho:",a,b,c,d,rho)
    print("beta1, beta2", beta_1, beta_2)
    print("P_max",P_max)
    
    #P_G = cp.Variable()
    #objective = cp.Minimize(((a+b+c/P_G+2*(a*b+a*c/P_G)**0.5)/(P_max-P_G)+beta_1/(2*beta_2)) / (1-2*beta_2*(rho+d/P_G)))
    #constraints = [
    #    (1 - 2 * beta_2 * rho) / (2 * beta_2 * d) <= P_G,
    #    P_G <= P_max 
    #]
    #problem = cp.Problem(objective, constraints)
    #problem.solve()
    def objective(P_G):
        return ((a+b+c/P_G+2*(a*b+a*c/P_G)**0.5)/(P_max-P_G)+beta_1/(2*beta_2)) / (1-2*beta_2*(rho+d/P_G))
    #x = np.linspace((1 - 2 * beta_2 * rho) / (2 * beta_2 * d) + 0.0001, P_max-0.0001 , 100)
    #y = objective(x)
    #plt.plot(x, y)
    #plt.savefig('./obj.png')
    left = (2 * beta_2 * d)/(1 - 2 * beta_2 * rho)
    right = P_max
    
    print("min:", left)
    print("max:", right)
    
    P_G = Golden_Search(objective,left,right)
    print("proposed:",objective(P_G), "even:", objective(P_max/3))

    #P_G = P_G.value
    P_u = (P_max - P_G)*((a*P_G)**0.5/((a*P_G)**0.5 + (b*P_G+c)**0.5))
    P_v = P_max - P_G - P_u

    print(P_u, P_v, P_G)

    return P_u, P_v, P_G, user_list, objective(P_G), objective(P_max/3), objective(0.9*P_max)

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

if __name__ == '__main__':
    args = args_parser()
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dict_users = iid(dataset_train, args.num_users)

    D = 20000
    v_max = 0.4
    u_max = 0.1
    #P_max = 1000
    beta_1 = 3.0
    beta_2 = 4.0

    # communication settings
    sigma_n = 1e-5 # sqrt(noise power)
    PL_exponent = 2.5 # User-BS Path loss exponent
    fc = 915 * 10**6 # carrier frequency 915MHz
    wave_lenth = 3.0 * 10**8 / fc # wave_lenth = c/f
    BS_gain = 10**(15.0/10) # BS antenna gain 20dBi
    User_gain = 10**(5.0/10) # user antenna gain 5dBi
    dist_max = 1000.0 # to quantify distance
    BS_hight = 10 # BS hight is 10m

    # optimal F is comming soon, now we just don't consider it
    F = [1.0 / np.sqrt(args.N)] * args.N

    if args.radius == 'same': 
        radius = np.array([50] * args.num_users) # same distance of all users
    elif args.radius == 'random_small':
        radius = np.random.rand(args.num_users) * 25 + 25 # (25,50)
    elif args.radius == 'random_large':
        radius = np.concatenate(np.random.rand(args.num_users-int(np.round(args.num_users/2))) * 25 + 25, int(np.round(args.num_users/2)) * 50 + 150) # half (25,50) and half (150,200)
    else:
        exit('Error: unrecognized radius')

    # get distances from BS to Users
    BS_dist = np.sqrt(radius**2 + BS_hight**2)

    # get path loss
    Path_loss = BS_gain * User_gain * (wave_lenth / (4 * np.pi * BS_dist))**PL_exponent

    # get shadow loss
    #shadow_loss = (np.random.randn(args.N, args.num_users) + 1j * np.random.randn(args.N, args.num_users)) / 2**0.5
    shadow_loss = np.ones((args.N, args.num_users))

    print("PL", Path_loss)

    H_origin = shadow_loss * np.sqrt(Path_loss)

    print(H_origin)
    # get the number of dataset per user have

    #exit('test')
    data_per_user = []
    for _,v in dict_users.items():
        data_per_user.append(len(v))

    proposed_list = []
    even_list = []
    main_grad_list = []

    """ for P_max in range(20, 501, 5):
        _, _, _,_, proposed, even, main_grad = optimal_power(P_max, beta_1, beta_2, data_per_user, F, H_origin, u_max, v_max, D, sigma_n)
        proposed_list.append(proposed)
        even_list.append(even)
        main_grad_list.append(main_grad)
    
    plt.plot(range(20, 501, 5), proposed_list, label='Proposed')
    plt.plot(range(20, 501, 5), even_list, label='Even')
    plt.plot(range(20, 501, 5), main_grad_list, label='GradientPrimary')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Objective with different power allocations')
    plt.xlabel('total power')
    plt.ylabel('Objective function')

    # 显示网格
    plt.grid(True)

    # 展示图像
    #plt.show()
    plt.savefig('simulation_results/obj_power_{}.png'.format(datetime.now().strftime('%Y_%m_%d_%H:%M:%s'))) """
    P_max = 100
    
    for distance in range(20, 150, 2):
        radius = np.array([distance] * args.num_users)
        BS_dist = np.sqrt(radius**2 + BS_hight**2)
        Path_loss = BS_gain * User_gain * (wave_lenth / (4 * np.pi * BS_dist))**PL_exponent
        H_origin = shadow_loss * np.sqrt(Path_loss)
        _, _, _,_, proposed, even, main_grad = optimal_power(P_max, beta_1, beta_2, data_per_user, F, H_origin, u_max, v_max, D, sigma_n)
        proposed_list.append(proposed)
        even_list.append(even)
        main_grad_list.append(main_grad)

    plt.plot(range(20, 150, 2), proposed_list, label='Proposed')
    plt.plot(range(20, 150, 2), even_list, label='Even')
    plt.plot(range(20, 150, 2), main_grad_list, label='GradientPrimary')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Objective with different distances')
    plt.xlabel('distances')
    plt.ylabel('Objective function')

    # 显示网格
    plt.grid(True)

    # 展示图像
    #plt.show()
    plt.savefig('simulation_results/obj_dis_{}.png'.format(datetime.now().strftime('%Y_%m_%d_%H:%M:%s')))