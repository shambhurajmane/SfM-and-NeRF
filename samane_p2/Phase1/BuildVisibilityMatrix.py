from Wrapper import *
import numpy as np
import math
import random
import matplotlib.pyplot as plt


def build_visibility_matrix(final_camera_poses, camera_matrix, depth_points, images):

    n_cameras = len(final_camera_poses)
    camera_indices = []
    point_indices = []
    points_2d = []
    points_3d = []
    points_count = 0
    for i in range(n_cameras):
        for j in range(len(depth_points)):
            for k in range(len(depth_points[j])):
                R = final_camera_poses[i][1]
                t = final_camera_poses[i][0]
                transformed_points = np.dot(R, depth_points[j][k] - t)

                # Project points onto the image plane
                projected_points = np.dot(camera_matrix, transformed_points)
                projected_points /= projected_points[2]

                # Check visibility and update matrix
                if projected_points[0] >= 0 and projected_points[0] < camera_matrix[0, 2]*2 and projected_points[1] >= 0 and projected_points[1] < camera_matrix[1, 2]*2:
                    camera_indices.append(i)
                    point_indices.append(points_count)
                    points_2d.append(projected_points[:2])
                    points_3d.append(depth_points[j][k])
                    points_count += 1
    
    return n_cameras, points_count, np.array(camera_indices), np.array(point_indices), np.array(points_2d), np.array(points_3d)