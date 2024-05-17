# 只需改如下四个参数即可
def Golden_Search(f, l, r, tol=1e-10, max_iter=10000):
    a = l
    b = r

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

def f(x):
    return (x-2.37)**2

print(Golden_Search(f,0,4))