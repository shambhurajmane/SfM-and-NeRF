import numpy as np
import random
import cv2
import ipdb
from EstimateFundamentalMatrix import find_fundamental_mat

def ransac_wh(keypoints1, keypoints2, matched_features, num_iterations, threshold):

    # Perform RANSAC to find the best homography
    H_matrix = None
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matched_features]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matched_features]).reshape(-1, 1, 2)
    filtered_matches = []
    count = 0 
    final_matches = None
    for i in range(num_iterations):
        # Randomly select 4 matched features
        try:
            random_matches = random.sample(matched_features, 4)
            random_points1 = np.float32([keypoints1[m.queryIdx].pt for m in random_matches]).reshape(-1, 1, 2)
            random_points2 = np.float32([keypoints2[m.trainIdx].pt for m in random_matches]).reshape(-1, 1, 2)

            # Find the homography matrix
            H, mask = cv2.findHomography(random_points1, random_points2)
            inliers, filtered_matches, matching_pairs = find_inliers(points1, points2, H,threshold)
            # ipdb.set_trace()    
            if inliers > count:
                # filtered_matches.append(random_matches)
                count = inliers
                H_matrix = H
                final_matches = filtered_matches   
                final_matching_pairs = matching_pairs
        except UnboundLocalError or ValueError:
            print("Not enough matches found")
            H_matrix = None
            final_matches =None
            

    print("RANSAC performed")
    print("Number of inliers", count)
    return H_matrix, final_matches, final_matching_pairs


def ransac_wa(keypoints1, keypoints2, matched_features, num_iterations, threshold):

    # Perform RANSAC to find the best homography
    A_matrix = None
    matching_pairs = np.float32([[[keypoints1[m.queryIdx].pt[0],keypoints1[m.queryIdx].pt[1]],[keypoints2[m.queryIdx].pt[0],keypoints2[m.queryIdx].pt[1]]] for m in matched_features])

    filtered_matches = []
    final_matching_pairs=[]
    count = 0 
    final_matches = None
    for i in range(num_iterations):
        # Randomly select 8 matched features
        try:
            random_matches = random.sample(matched_features, 8)
            random_pairs = np.float32([[[keypoints1[m.queryIdx].pt[0],keypoints1[m.queryIdx].pt[1]],[keypoints2[m.queryIdx].pt[0],keypoints2[m.queryIdx].pt[1]]] for m in random_matches])

            # Find the homography matrix
            A = find_fundamental_mat(random_pairs)
            inliers, filtered_matches, matching_pairs = find_inliers_wa(matching_pairs, A,threshold)
            # ipdb.set_trace()    
            if inliers > count:
                # filtered_matches.append(random_matches)
                count = inliers
                A_matrix = A
                final_matches = filtered_matches   
                final_matching_pairs = matching_pairs
        except UnboundLocalError or ValueError:
            print("Not enough matches found")
            A_matrix = None
            final_matches =None
            

    print("RANSAC performed")
    print("Number of inliers", count)
    return A_matrix, final_matches, final_matching_pairs

def find_inliers(src_points, dst_points, H1, threshold):
    number_of_inliers = 0
    filtered_matches = []  
    matching_pairs = [] 
    for i in range(len(src_points)):
        src_point = [src_points[i,0][0], src_points[i,0][1]] 	
        src_point.append(1)
        src_point = np.array(src_point)
        src_point = src_point.reshape(3,1)

        dest_point = [dst_points[i,0][0], dst_points[i,0][1]] 	
        dest_point.append(1)
        dest_point = np.array(dest_point)
        dest_point = dest_point.reshape(3,1)

        predicted_point = np.dot(H1, src_point)	
        # calculate the sum of squared distance between the predicted
        ssd = np.linalg.norm(dest_point[0:1] - predicted_point[0:1])
        # ipdb.set_trace()		
        if ssd < threshold:
            filtered_matches.append(cv2.DMatch(i, i, 0))
            matching_pairs.append([[src_points[i,0][0], src_points[i,0][1]],[dst_points[i,0][0], dst_points[i,0][1]]])
            number_of_inliers += 1
 
    return number_of_inliers , filtered_matches ,matching_pairs


def find_inliers_wa(matching_pairs, A,threshold):
    number_of_inliers = 0
    filtered_matches = []  
    final_matching_pairs = [] 
    for i in range(len(matching_pairs)):
        left_pt = matching_pairs[i][0].tolist()
        np.array(left_pt.append(1))
        right_pt = matching_pairs[i][1].tolist()
        np.array(right_pt.append(1))

        epipolar_constraint = np.matmul(left_pt,A)
        epipolar_constraint = np.matmul(epipolar_constraint,right_pt)
        # ipdb.set_trace()
        if abs(epipolar_constraint) < threshold:
            filtered_matches.append(cv2.DMatch(i, i, 0))
            final_matching_pairs.append(matching_pairs[i])
            number_of_inliers += 1
 
    return number_of_inliers , filtered_matches ,final_matching_pairs


# #without uncertainty or randomness
# def ransac_wa(keypoints1, keypoints2, matched_features, num_iterations, threshold):

#     # Perform RANSAC to find the best homography
#     A_matrix = None
#     matching_pairs = np.float32([[[keypoints1[m.queryIdx].pt[0],keypoints1[m.queryIdx].pt[1]],[keypoints2[m.queryIdx].pt[0],keypoints2[m.queryIdx].pt[1]]] for m in matched_features])

#     filtered_matches = []
#     final_matching_pairs=[]
#     count = 0 
#     final_matches = None
#     for i in range(len(matched_features)-8):
#         # Randomly select 8 matched features
#         try:
#             random_matches = []
#             add= 0
#             for count in range(8):
#                 random_matches.append(matched_features[i+add])
#                 add+=1
#             random_pairs = np.float32([[[keypoints1[m.queryIdx].pt[0],keypoints1[m.queryIdx].pt[1]],[keypoints2[m.queryIdx].pt[0],keypoints2[m.queryIdx].pt[1]]] for m in random_matches])

#             # Find the homography matrix
#             A = find_fundamental_mat(random_pairs)
#             inliers, filtered_matches, matching_pairs = find_inliers_wa(matching_pairs, A,threshold)
#             # ipdb.set_trace()    
#             if inliers > count:
#                 # filtered_matches.append(random_matches)
#                 count = inliers
#                 A_matrix = A
#                 final_matches = filtered_matches   
#                 final_matching_pairs = matching_pairs
#         except UnboundLocalError or ValueError:
#             print("Not enough matches found")
#             A_matrix = None
#             final_matches =None
            

#     print("RANSAC performed")
#     print("Number of inliers", count)
#     return A_matrix, final_matches, final_matching_pairs