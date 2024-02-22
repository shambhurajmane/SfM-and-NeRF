import numpy as np
import random
import cv2
import ipdb
import matplotlib.pyplot as plt
from scipy.linalg import rq
from LinearPnP import *

def PnP_RANSAC(K, x_2D, x_3D, iterations, threshold):

    camera_pose = None
    num_points = x_2D.shape[0]
    u = x_2D[:, 0]
    v = x_2D[:, 1]

    ones = np.ones((x_3D.shape[0], 1))
    world_points_homogen = np.concatenate((x_3D, ones), axis=1)

    
    final_matches = []
    Prrdddd = []

    for index in range(iterations):                                       
        count = 0 
        filtered_matches = []

        random_indeces = random.sample(range(num_points - 1), 6)

        random_2d_points = x_2D[random_indeces, :]
        random_3d_points = x_3D[random_indeces, :]

        Projection_left = linear_PnP(K, random_2d_points, random_3d_points)
        proj = np.matmul(Projection_left, world_points_homogen.T)
        proj = proj / proj[-1]
        proj = proj.T

        for i in range(len(proj)):
            error = np.sqrt((u[i] - proj[i][0])**2 + (v[i] - proj[i][1])**2)
            if error < threshold:
                count += 1
                filtered_matches.append(i)

        if len(final_matches) < count:
            final_matches = filtered_matches
            count = len(filtered_matches)
            Prrdddd.append(Projection_left)

    print("count", count) 

    feature_inliers = x_2D[final_matches]
    world_point_inliers = x_3D[final_matches]

    Prr = linear_PnP(K, feature_inliers, world_point_inliers)
    proj = np.matmul(Prr, world_points_homogen.T)
    proj = proj / proj[-1]
    proj = proj.T

    for i in range(len(proj)):
        error = np.sqrt((u[i] - proj[i][0])**2 + (v[i] - proj[i][1])**2)
    d1= np.array([0,0,0])
    Kr=np.append(K,d1.reshape((3, 1)) , axis=1)
    R_camera = Prr[0:3, 0:3]
    K_inv = np.linalg.inv(K)
    R = K_inv @ R_camera

    U_R, D_R, V_T_R = np.linalg.svd(R)                 
    R = U_R @ V_T_R                                   

    lamda = D_R[0]                              

    t = Prr[:, 3]
    T = K_inv @ t / lamda

    R_det = np.linalg.det(R)

    if R_det < 0:
        R = -R
        T = -T

    C = -R.T @ T

    camera_pose = [ C , R]

    return camera_pose,[Kr,Prr]

