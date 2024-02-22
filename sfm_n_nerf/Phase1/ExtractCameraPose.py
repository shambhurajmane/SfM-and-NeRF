import numpy as np
import math
import random
import matplotlib.pyplot as plt


def extract_camera_pose(E_matrix):
    # Create a matrix to store the coefficients of the system of linear equations
    camera_poses = []   # C,R
    U, S, V = np.linalg.svd(E_matrix)
    W_matrix = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    C1 = U[:,-1]
    R1 = U @ W_matrix @ V
    #Empose the constraint of orthogonal matrix
    U1,S1,V1 = np.linalg.svd(R1)
    R1 = U1 @ V1
    R1 = np.array(R1)
    camera_poses.append([C1,R1])
    camera_poses.append([-C1,R1])


    R2 = U @ W_matrix.T @ V
    #Empose the constraint of orthogonal matrix
    U2,S2,V2 = np.linalg.svd(R2)
    R2 = U2 @ V2
    R2 = np.array(R2)
    camera_poses.append([C1,R2])
    camera_poses.append([-C1,R2])

    C_list, R_list = [], []

    for i in  range(len(camera_poses)):
        if np.linalg.det(camera_poses[i][1]) < 0:
            camera_poses[i] =  [-camera_poses[i][0],-camera_poses[i][1]] 


    return camera_poses