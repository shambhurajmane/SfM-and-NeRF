"""
Computing the best fundamental matrix from the matching features using RANSAC
"""

import numpy as np
from EstimateFundamentalMatrix import *
import ipdb
import random


def get_inliers_RANSAC(matched_points,iterations,threshold):
    num_matches = len(matched_points)              
    A_matrix = np.zeros((3, 3))                    

    number_of_inliers = 0    
    filnal_matches = []                            

    for index in range(iterations):               

        # Select 8 matched feature pairs from each image at random
        points = [np.random.randint(0, num_matches) for num in range(8)]      
        left_pts = []
        right_pts = []
        for pt in points:
            left_pts.append(matched_points[pt, 0])
            right_pts.append(matched_points[pt, 1])

        left_pts = np.array(left_pts, np.float32)
        right_pts = np.array(right_pts, np.float32)

        F = get_fundamental_matrix(left_pts, right_pts)      
        num_filtered_matches = 0

        filtered_matches = []

        for i in range(num_matches):

            ul = matched_points[i, 0, 0]
            vl = matched_points[i, 0, 1]
            ur = matched_points[i, 1, 0]
            vr = matched_points[i, 1, 1]

            # Homogeneous coordinates
            point = np.array([ul, vl, 1], np.float32)
            point_prime = np.array([ur, vr, 1], np.float32)

            # Applying epipolar constraint to the points to check if they are inliers
            epipolar_constraint = np.matmul(F, point.T)
            epipolar_constraint = np.multiply(point_prime, epipolar_constraint.T)

            error = np.sum(epipolar_constraint)
            if abs(error) < threshold:
                num_filtered_matches += 1
                match_f = [[ul, vl], [ur, vr]]
                filtered_matches.append(match_f)

        if number_of_inliers < num_filtered_matches:
            number_of_inliers = num_filtered_matches
            filtered_matches = np.asarray(filtered_matches)
            matches_p = filtered_matches[:, 0]
            matches_p_prime = filtered_matches[:, 1]

            # Compute the fundamental matrix for the matched pairs
            A_matrix = get_fundamental_matrix(matches_p, matches_p_prime)

            # Set for the output array of best matched points
            filnal_matches = filtered_matches


    return A_matrix, filnal_matches


def visualize_matches(image1, image2, matched_points):

    # Handling the case of a color or greyscale image passed to the visualizer
    if len(image1.shape) == 3:
        height1, width1, depth1 = image1.shape
        height2, width2, depth2 = image2.shape
        shape = (max(height1, height2), width1 + width2, depth1)

    elif len(image1.shape) == 2:
        height1, width1, depth1 = image1.shape
        height2, width2, depth2 = image2.shape
        shape = (max(height1, height2), width1 + width2)

    image_combined = np.zeros(shape, type(image1.flat[0]))  # blank side-by-side images
    image_combined[0:height1, 0:width1] = image1  # fill in left side image
    image_combined[0:height1, width1:width1 + width2] = image2  # fill in right side image
    image_1_2 = image_combined.copy()

    circle_size = 4
    red = [0, 0, 255]
    cyan = [255, 255, 0]
    yellow = [0, 255, 255]

    # Mark each matched corner pair with a circle and draw a line between them
    for i in range(len(matched_points)):
        corner1_x = matched_points[i][0][0]
        corner1_y = matched_points[i][0][1]
        corner2_x = matched_points[i][1][0]
        corner2_y = matched_points[i][1][1]

        cv2.line(image_1_2, (int(corner1_x), int(corner1_y)), (int(corner2_x + image1.shape[1]), int(corner2_y)), red,
                 1)
        cv2.circle(image_1_2, (int(corner1_x), int(corner1_y)), circle_size, cyan, 1)
        cv2.circle(image_1_2, (int(corner2_x) + image1.shape[1], int(corner2_y)), circle_size, yellow, 1)

    # Resize for better displaying
    scale = 1.5
    height = image_1_2.shape[0] / scale
    width = image_1_2.shape[1] / scale
    im = cv2.resize(image_1_2, (int(width), int(height)))

    # Save as "".png
    cv2.imwrite('Filtered matches' + '.png', im)
    cv2.imshow("Matches visualizing", im)
    cv2.waitKey(0)

