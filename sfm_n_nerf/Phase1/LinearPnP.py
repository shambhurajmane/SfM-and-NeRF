"""
persepctive n point 
"""
import random
import numpy as np


def linear_PnP(K, x_2D, x_3D):
    X = x_3D[:, 0]
    Y = x_3D[:, 1]
    Z = x_3D[:, 2]

    zeros = np.zeros_like(X)          # vector of ones the length of x_3D
    ones = np.ones_like(X)            # vector of zeros the length of x_3D

    x_2D = np.concatenate((x_2D, ones.reshape(len(x_2D),1)), axis=1)  # add a column of ones to the 2D points
                
    u = x_2D[:, 0]
    v = x_2D[:, 1]

    A1 = np.vstack([-X, -Y, -Z, -ones, zeros, zeros, zeros, zeros, u * X, u * Y, u * Z, u]).T
    A2 = np.vstack([zeros, zeros, zeros, zeros, -X, -Y, -Z, -ones, v * X, v * Y, v * Z, v]).T
    A = np.vstack([A1, A2])

    # Perform SVD on the system of equations
    U, D, V_T = np.linalg.svd(A)                        # output matrix V_T is shape [12 x 12]
    P = V_T[-1, :]                                      # -1 refers to the last element, the maximum index number
    Prr = P.reshape((3, 4))                               # reshape to form the projection matrix P [3 x 4]

    return Prr