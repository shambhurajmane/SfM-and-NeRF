import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import random


def find_essential_mat(F_matrix, calib_mat):

    # Create a matrix to store the coefficients of the system of linear equations
    E = np.matmul(calib_mat.T,np.matmul(F_matrix,calib_mat))
    # Perform singular value decomposition
    U,S,V = np.linalg.svd(E)
    sigma_matrix = np.diag([1,1,0])
    
    E = np.matmul(U,np.matmul(sigma_matrix,V))

    return E


