import numpy as np
import matplotlib.pyplot as plt
import random
import math


def disambiguate_camera_poses(camera_poses, Xr_points):

    num_poses = len(camera_poses)
    total_count = []               

    for i in range(num_poses):                          
        R = camera_poses[i][1]                                 
        t = camera_poses[i][0]                                  
        Xr_point = Xr_points[i]                    
        r_3 = R[:, 2]                   

        count = 0                                
        for pt in range(len(Xr_point)):                  
            X = Xr_point[pt]                          
            z = X[2]                                 
            cond_1 = r_3.T @ (X.T - t) > 0                  # The cheirality condition
            cond_2 = z > 0                                  # is Z point positive (in front of image plane)

            if cond_1 and cond_2:
                count = count + 1                          
        total_count.append(count)                  

    total_count = np.array(total_count)                   
    id = np.argmax(total_count)                     

    correct_pose = camera_poses[id]                            
    X_correct = Xr_points[id]          

    return correct_pose , X_correct, id






    

        
        
