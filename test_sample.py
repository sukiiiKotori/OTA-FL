import random
import numpy as np

def allocate_datasets(m, n):
    # 检查是否有足够的数据分配给所有用户
    if n > m:
        raise ValueError("无法为每个用户分配唯一大小的数据集，因为用户数大于数据总量")
    
    # 生成n-1个随机数
    random_points = random.sample(range(1, m), n-1)
    
    # 对随机数进行排序
    random_points.sort()
    
    # 计算相邻随机数的差值，即每个用户分配的数据集大小
    dataset_sizes = [random_points[0]] + [random_points[i+1] - random_points[i] for i in range(n-2)]
    print([random_points[0]])
    print([random_points[i+1] - random_points[i] for i in range(n-2)])
    
    # 添加最后一个用户的数据集大小
    dataset_sizes.append(m - random_points[-1])
    
    # 打乱列表顺序
    random.shuffle(dataset_sizes)
    
    return dataset_sizes

# 测试代码
m = 60000  # 数据总量
n = 60   # 用户数
#print(allocate_datasets(m, n))

sizes = np.random.randint(1, m, size=n-1)
sizes.sort()
sizes = np.append(sizes,1)
print(list(set([1,2,3,4,5,6,7,8])-[5,8]))