from models.Nets import CNNMnist
from utils.options import args_parser
from collections import OrderedDict
import numpy as np
import copy
import torch

""" def get_args(var_receive, num_dataset, p_g, F, H):  
    inner = np.array(F).conj() @ H #(1xN x NxM = 1xM)
    print(inner)
    print(inner.shape)
    eta_arr = (np.abs(inner) / (np.array(num_dataset) * np.array(var_receive)))**2
    print(eta_arr)
    eta = p_g * np.min(eta_arr)
    p_gm = np.array(num_dataset) * np.sqrt(eta) * np.array(var_receive) / inner
    return p_gm, eta

H = np.ones((5, 3), dtype = complex)

H_2 = np.ones((5,1), dtype= complex)

H_3 = np.array([1,2,3,4,5])

print(H+H_3.reshape(-1,1))
print(H * H_3.reshape(-1,1))
#print(H)
F = [1.0,0,0,0,0]

var_receive = [2.5,4.3,3.2]
num_data = [100,150,300]

print(get_args(var_receive,num_data,1,F,H))

nparr1 = np.array(var_receive)
nparr2 = np.array(num_data)"""

net1 = CNNMnist(args_parser())

weight = net1.state_dict()

flat_grad = np.array([])
for _, v in weight.items():
    flat_grad = np.hstack((flat_grad, v.numpy().ravel()))

temp_dict = OrderedDict()

idx = 0
for k in weight.keys():
    shape = np.array(weight[k].size())
    if len(shape):
        lenth = np.prod(shape)
        temp = weight[k] - torch.from_numpy(np.reshape(flat_grad[idx:idx+lenth],shape)).float() *1.0
        idx += lenth
        temp_dict[k] = temp

print(temp_dict)
print('------------------------------------------------------------------------------------------------------------------------------------')
#print(dict)

"""
weight_list = [weight,weight2]
print(type(weight_list))

t = np.array([np.pi,2*np.pi,1.5*np.pi,1.1])
print(t)
print(np.exp(1j * t))

print([0]*5)

#print(weight)

for k in weight.keys():
    weight[k] += 0.1

#print(weight)

test_n = (np.random.randn(5, 3)+1j*np.random.randn(5, 3)) / 2**0.5

print(test_n.shape)
print(np.array(F).conj() @ test_n)

var_locals = [1+0.5j,2,3+0.1j,4,5]

mean_locals = [-0.5, 1.2, 3.3,-2.1,-1.3]

temp = np.array(var_locals)*np.sign(np.array(mean_locals))

print(type(np.array(temp)))
print(type(temp))
print(type(list(temp)))
print(np.abs(temp))
print(list(np.sign(np.real(temp)))) """

""" def transmit_grad(grad_locals, mean_locals, var_locals, p_gm):
    # apply formula (9)
    normal_grad =  (grad_locals - np.array(mean_locals).reshape(-1,1)) / np.array(var_locals).reshape(-1,1)
    return np.array(p_gm).reshape(-1,1) * normal_grad

grad_locals = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
mean_locals = np.array([0.5,0.5,0.5,0.5,0.5])
var_locals = np.array([2,2,2,2,2])
p_gm = np.array([1,1,1,1,1])
t = 1 + 1j
print(np.real(t))

print(transmit_grad(grad_locals,mean_locals,var_locals,p_gm)) """