import numpy as np
import random
import cv2
import ipdb
import matplotlib.pyplot as plt




def find_initial_traingulation(camera_poses, matching_points, calib_mat):
    # Create a matrix to store the coefficients of the system of linear equations
    cheirality_count = [] 
    for c,camera_pose in enumerate(camera_poses):
        t = camera_pose[0]
        r = camera_pose[1]
        print(t,r)
        Tr = np.append(r,t,axis=1)
        p = np.matmul(calib_mat,Tr)
        m = np.append(calib_mat,[[0],[0],[0]],axis=1)
        curr_count = 0
        
        for i in range(len(matching_points)):
            ul, vl = matching_points[i][0]
            # ipdb.set_trace()
            ur, vr = matching_points[i][1]
            A = np.zeros((4, 3))
            b = np.zeros((4, 1))

            # Add the coefficients to the matrix

            A[0] = [ur*m[2,0] - m[0,0], ur*m[2,1] - m[0,1], ur*m[2,2] - m[0,2]]

            A[1] = [vr*m[2,0] - m[1,0], vr*m[2,1] - m[1,1], vr*m[2,2] - m[1,2]]
            A[2] = [ul*p[2,0] - p[0,0], ul*p[2,1] - p[0,1], ul*p[2,2] - p[0,2]]
            A[3] = [vl*p[2,0] - p[1,0], vl*p[2,1] - p[1,1], vl*p[2,2] - p[1,2]]

            b[0] = [m[0,3]-m[2,3]]
            b[1] = [m[1,3]-m[2,3]]
            b[2] = [p[0,3]-p[2,3]]
            b[3] = [p[0,3]-p[2,3]]

            Xr = np.matmul(np.linalg.inv(np.matmul(A.T,A)),np.matmul(A.T,b))
            colors= ["red", "blue", "green", "orange"]
            plt.plot(vr,Xr[2][0],marker="o", color=colors[c])

            r3 = camera_pose[1][2,:]
            C = camera_pose[0]
            if np.matmul(r3,(Xr-C)) > 0:
                curr_count+=1
        cheirality_count.append(curr_count)

        plt.xlabel("ur")
        plt.xlabel("Xr")

        plt.show()
    index= cheirality_count.index(max(cheirality_count))

    return camera_poses[index]

