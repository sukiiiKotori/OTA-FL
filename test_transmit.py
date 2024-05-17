import numpy as np
from AirComp import *

# 测试用例
def test_transmit_grad():
    # 创建测试数据
    m = 5  # 客户端数量
    D = 10  # 模型参数维度
    grad_locals = np.random.randn(m, D)  # 随机生成梯度矩阵
    mean_locals = np.random.randn(m)  # 随机生成均值向量
    var_locals = np.random.rand(m)  # 随机生成方差向量
    p_gm = np.random.rand(m)  # 随机生成权重向量

    # 调用函数
    weighted_grad = transmit_grad(grad_locals, mean_locals, var_locals, p_gm)
    print(grad_locals)
    print(weighted_grad)

    # 验证输出形状
    assert weighted_grad.shape == (m, D), "Output shape mismatch"

    # 验证每个客户端的梯度是否被正确标准化和加权
    for i in range(m):
        normal_grad_i = (grad_locals[i] - mean_locals[i]) / var_locals[i]
        weighted_grad_i = weighted_grad[i]
        assert np.allclose(weighted_grad_i, p_gm[i] * normal_grad_i), "Gradient not correctly normalized or weighted"

    print("Test passed!")

if __name__ == "__main__":
    test_transmit_grad()