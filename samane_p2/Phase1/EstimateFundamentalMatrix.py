import numpy as np
import random
import cv2
import ipdb

def get_fundamental_matrix(left_points, right_points):

    vec_one = np.ones(left_points.shape[0])

    ul, vl, ur, vr = left_points[:, 0], left_points[:, 1], right_points[:, 0], right_points[:, 1]

    A = np.asarray([ul * ur, vl * ur, ur, ul * vr, vl * vr, vr, ul, vl, vec_one])  
    A = np.transpose(A)                                

    # Perform SVD
    U, S, V = np.linalg.svd(A)                        
    f = V[-1, :]                                     

    f = f.reshape(3, 3)                               

    U, S, V = np.linalg.svd(f)

    S[2] = 0.0                                   
    sigma_mat = np.diag(S)                           
    F = np.matmul(U,np.matmul(sigma_mat,V))              

    return F