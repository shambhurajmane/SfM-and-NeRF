import numpy as np
import random
import cv2
import ipdb



def find_camera_pose(E_matrix):
    # Create a matrix to store the coefficients of the system of linear equations
    camera_poses = []   # C,R
    U, S, v = np.linalg.svd(E_matrix)
    W_matrix = np.matrix([[0,-1,0],[1,0,0],[0,0,1]])

    
    R1 = np.matmul(U,np.matmul(W_matrix,v))
    detR1= np.linalg.det(R1)
    C1 = U[:,-1]
    if detR1 <0:
        camera_poses.append([-C1,-R1])
        camera_poses.append([C1,-R1])
    else:
        camera_poses.append([C1,R1])
        camera_poses.append([-C1,R1])
    R2 = np.matmul(U,np.matmul(W_matrix.T,v))
    detR2= np.linalg.det(R2)
    if detR2 <0:
        camera_poses.append([-C1,-R2])
        camera_poses.append([C1,-R2])
    else:
        camera_poses.append([C1,R2])
        camera_poses.append([-C1,R2])
    


    return camera_poses

