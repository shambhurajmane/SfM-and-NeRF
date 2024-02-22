import numpy as np
import random
import cv2
import ipdb
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def non_linear_triangulation( filtered_matched_points, world_points,projections):
    
    optimized_wp = []
    for i in range(len(world_points)):
        left_pt = filtered_matched_points[i][0]
        right_pt = filtered_matched_points[i][1]  
        ur, vr, wr = world_points[i]
        wp = np.array([ur, vr, wr])

        optimum_wp = least_squares(lambda x: error_function(x, left_pt,right_pt, projections[0], projections[1], i), x0=wp, method='trf')
        optimum_wp = optimum_wp.x
        
        optimized_wp.append(optimum_wp)
    return optimized_wp

    
def error_function(x, left_pt,right_pt, pr, pl, id):
    x=x.tolist()
    x.append(1)
    x=np.array(x)

    left_proj = np.matmul(pr,x)
    left_proj = left_proj / left_proj[-1]

    right_proj = np.matmul(pl,x)
    right_proj = right_proj / right_proj[-1]
    L_error_reproj = (left_proj[0:2] - left_pt) ** 2
    R_error_reproj = (right_proj[0:2] - right_pt) ** 2

    return np.concatenate((R_error_reproj,L_error_reproj))