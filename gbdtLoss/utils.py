import numpy as np

def sigmoid(x):
    # Reference: https://github.com/Luojiaqimath/ClassbalancedLoss4GBDT/blob/main/Examples/binary_lgb.py#L13
    kEps = 1e-16 #  avoid 0 div
    x = np.minimum(-x, 88.7)  # avoid exp overflow
    return 1 / (1 + np.exp(x)+kEps)

def check_gradient(func, grad, values, eps=1e-8):
    approx = (func(values + eps) - func(values - eps)) / (2 * eps)
    return np.linalg.norm(approx - grad(values))