import numpy as np
import random
import cv2
import ipdb
import matplotlib.pyplot as plt



def find_linear_traingulation(Camera_mat, camera_pose, best_matched_points):

    left_point = best_matched_points[:, 0]
    right_point = best_matched_points[:, 1]

    ones = np.ones((left_point.shape[0], 1))

    points_1 = np.concatenate((left_point, ones), axis=1)
    points_2 = np.concatenate((right_point, ones), axis=1)

    t,R = camera_pose

    t = t.reshape((3, 1))                        
    Transformation_mat = np.append(R, -t, axis=1).tolist() 
    Transformation_mat.append([0,0,0,1])        
    Transformation_mat = np.array(Transformation_mat)

    d1= np.array([0,0,0])
    K=np.append(Camera_mat,d1.reshape((3, 1)) , axis=1)
    Projection_left = K @ Transformation_mat
    Projection_right= K

    X_pts = []
    num_points = best_matched_points.shape[0]

    for i in range(num_points):
        X_1_i = skew_matrix(points_1[i]) @ Projection_right
        X_2_i = skew_matrix(points_2[i]) @ Projection_left

        x_P = np.vstack((X_1_i, X_2_i))
        _, _, V_T = np.linalg.svd(x_P)
        X_pt = V_T[-1][:]

        X_pt = X_pt / X_pt[3]
   
        X_pt = X_pt[0:3]                   

        X_pts.append(X_pt)

    return X_pts , Projection_right, Projection_left

def skew_matrix(x):
    X = np.array([[0, -x[2] , x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    return X