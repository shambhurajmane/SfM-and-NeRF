import numpy as np
import random
import cv2
import ipdb



def find_fundamental_mat(matching_pairs):
    # Create a matrix to store the coefficients of the system of linear equations
    F = np.zeros((len(matching_pairs), 9))
    # For each pair of matching_pairs
    for i, [point1, point2] in enumerate(matching_pairs):
        ul, vl = point1
        ur, vr = point2

        # Add the coefficients to the matrix
        F[i] =  [ul*ur, ul*vr, ul, vl*ur, vl*vr, vl, ur, vr, 1]

    # Perform singular value decomposition
    U, S, f = np.linalg.svd(F)

    # The Fundamental matrix is the last column of f
    F = f[-1].reshape((3, 3))


    return F

def compute_fun_matrix(matching_pairs):
    # Create a matrix to store the coefficients of the system of linear equations
    F = np.zeros((len(matching_pairs), 9))
    # For each pair of matching_pairs
    for i, [point1, point2] in enumerate(matching_pairs):
        ul, vl = point1
        ur, vr = point2

        # Add the coefficients to the matrix
        F[i] =  [ul*ur, ul*vr, ul, vl*ur, vl*vr, vl, ur, vr, 1]

    # Perform singular value decomposition
    U, S, v = np.linalg.svd(F)
    S[-1]= 0 
    sigma_matrix = np.zeros((F.shape))
    for i, sigma_value in enumerate(S):
        sigma_matrix[i, i] = sigma_value
    
    F = np.matmul(U,np.matmul(sigma_matrix,v))
    U, S, f = np.linalg.svd(F)

    # The Fundamental matrix is the last column of f
    F = f[-1].reshape((3, 3))

    return F
