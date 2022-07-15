import numpy as np
import math
def define_SE3(R, p):
    SE3 = np.identity(4)
    SE3[0:3, 0:3] = R
    SE3[0:3, 3] = p
    return SE3

def get_SO3(SE3):
    return SE3[0:3, 0:3]

def get_p(SE3):
    return SE3[0:3, 3]

def change_SO3(SE3, R):
    SE3[0:3, 0:3] = R
    return SE3

def change_p(SE3, p):
    SE3[0:3, 3] = p
    return SE3

def inverse_SE3(SE3):
    R = np.transpose(get_SO3(SE3))
    p = - np.dot(R, get_p(SE3))
    inv_SE3 = define_SE3(R, p)
    return inv_SE3

def transform_point(SE3, p):
    p_h = np.ones(4)
    p_h[0:3] = p
    p_h = np.dot(SE3, p_h)
    return p_h[0:3]

# skew matrix
def skew(w):
    W = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]])
    
    return W

# SO3 exponential
def exp_so3(w):
    if len(w) != 3:
        raise ValueError('Dimension is not 3')
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        R = np.eye(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        R = np.eye(3) + sw * wnorm_inv * W + (1 - cw) * np.power(wnorm_inv,2) * W.dot(W)

    return R