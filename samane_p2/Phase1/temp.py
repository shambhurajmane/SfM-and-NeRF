# Code starts here:

import ipdb
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import rotate
import os
import sklearn.cluster
from scipy import signal, ndimage
import random
from Wrapper import *
 
# Add any python libraries here
op_folder_name = "./Results"
if not os.path.exists(op_folder_name):
    os.makedirs(op_folder_name)
    
# To load images from given folder
def loadImages(folder_name, image_files):
	print("Loading images from ", folder_name)
	images = []
	if image_files is None:
		image_files = os.listdir(folder_name)
	for file in image_files:
		print("Loading image ", file, " from ", folder_name)
		image_path = folder_name + "/" + file
		
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if image is not None:
			images.append(image)
			
		else:
			print("Error in loading image ", image)

	return images


def create_visualization(images, Labels, cols, size,file_name):
	rows = int(np.ceil(len(images)/cols))
	plt.subplots(rows, cols, figsize=size)
	for index in range(len(images)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(images[index], cmap= "gray")
		plt.title(Labels[index])
	plt.savefig(file_name)	
	plt.show()
	plt.close()
	# saved_figure = cv2.imread(file_name)
	# return saved_figure

def create_image(image,file_name):
	plt.imshow(image, cmap='gray')
	plt.savefig(file_name)
	plt.close()
	plt.show()


def detect_corners(image, method):
    # Detect corners in the image
    # Inputs: image1 and image2 are two images to be stitched together
    # Output: detected_corners is the output after detecting corners in the image

    # Convert the images to grayscale
    temp = image.copy() 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    # Detect the corners in the image
    if method == "Harris":
        corners = cv2.cornerHarris(gray_image, 2, 3, 0.001)
        corners = cv2.dilate(corners, None)
        image[corners > 0.01 * corners.max()] = [0,0,255]
   
    elif method == "good_features_to_track":
        corners = cv2.goodFeaturesToTrack(gray_image, 1000, 0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(temp,(x,y),3,255,-1)
    return corners, temp
       

def anms(corners , method):  
    # Perform adaptive non-maximal suppression on the detected corners
    # Input: detected_corners is the output after detecting corners in the image
    # Output: anms_corners is the output after performing adaptive non-maximal suppression on the detected corners

    # Sort the corners based on the corner  strength
    if method == "Harris":
        corners = sorted(corners, key = lambda x:x[0][0], reverse=True) 
    elif method == "good_features_to_track":
        print("Not needed for good_features_to_track")
    return corners

def find_feature_descriptors(image, corners):
    # Find feature descriptors for the detected corners
    # Input: detected_corners is the output after detecting corners in the image
    # Output: descriptors is the output after finding the feature descriptors for the detected corners

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the feature descriptors for the detected corners
    patch_size = 41
    patch_size_half = int(patch_size/2)
    descriptors = []  
    keypoints = []    
    for corner in corners:
        x, y = corner.ravel()
        if x-patch_size_half-1 < 0 or y-patch_size_half-1 < 0 or x+patch_size_half+1 > image.shape[1] or y+patch_size_half+1 > image.shape[0]:
            continue
        patch = gray_image[y-patch_size_half:y+patch_size_half+1, x-patch_size_half:x+patch_size_half+1]
        patch = cv2.GaussianBlur(patch,(3,3),0)
        patch = cv2.resize(patch, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
        feature = patch.reshape(-1)
        feature = (feature-feature.mean())/ np.std(feature)
        descriptors.append(feature) 
        keypoints.append([x,y]) 

    return descriptors, keypoints

def match_features(descriptors1, descriptors2):
    # Match the features between the two images
    # Input: descriptors1 and descriptors2 are the feature descriptors for the detected corners in image1 and image2 respectively
    # Output: matched_features is the output after matching the features between the two images

    # Match the features between the two images
    matched_features = []
    for i, descriptor1 in enumerate(descriptors1):
        sum_square_dist = []
        for j, descriptor2 in enumerate(descriptors2):
            # ipdb.set_trace()
            sum_square_dist.append(np.sum((descriptor1 - descriptor2)**2))
        top_match1 = np.argmin(sum_square_dist)
        dist1 = sum_square_dist.pop(top_match1)
        top_match2 = np.argmin(sum_square_dist)
        dist2 = sum_square_dist.pop(top_match2)
        if dist1/dist2 < 0.5:
            matched_features.append(cv2.DMatch(i, top_match1, dist1)) 

    print("Matched features between the two images", len(matched_features))
    return matched_features


def ransac(keypoints1, keypoints2, matched_features, num_iterations, threshold):

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
            inliers, filtered_matches = find_inliers(points1, points2, H)
            # ipdb.set_trace()    
            if inliers > count:
                # filtered_matches.append(random_matches)
                count = inliers
                H_matrix = H
                final_matches = filtered_matches   
        except UnboundLocalError or ValueError:
            print("Not enough matches found")
            H_matrix = None
            final_matches =None
            

    print("RANSAC performed")
    print("Number of inliers", count)
    return H_matrix, final_matches

def find_inliers(src_points, dst_points, H1):
    number_of_inliers = 0
    filtered_matches = []   
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
        if ssd < 0.6:
            filtered_matches.append((src_point, dest_point))
            number_of_inliers += 1
 
    return number_of_inliers , filtered_matches

def warp_image(img1, img2, H):
    # Create a matrix of 4 points representing the four corners of the image to be warped, 
    #x is the width and y is the height here image shape (y,x)

    yp, xp = img1.shape[0], img1.shape[1] 
    Four_corners = np.matrix([[0,0,1],[0,yp,1],[xp,yp,1], [xp,0,1]])
    print(Four_corners)

    # Apply the transformation matrix
    transformed_points = np.matmul(H, Four_corners.T).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]

    transformed_points = transformed_points.T
    transformed_points = transformed_points.astype(int)

    minx, miny = np.min(transformed_points[1,:]),np.min(transformed_points[0,:])
    shiftx = -np.min([minx, 0]) 
    shifty = -np.min([miny, 0])
    maxx, maxy = np.max(transformed_points[1,:]), np.max(transformed_points[0,:])
    bottom_right_x = np.max([maxx+shiftx, img2.shape[0]+shiftx]) 
    bottom_right_y = np.max([maxy+shifty, img2.shape[1]+shifty])   
    canvas = np.zeros((bottom_right_x,bottom_right_y, 3), dtype=np.uint8)

    canvas[shiftx:img2.shape[0]+shiftx, shifty:img2.shape[1]+shifty] = img2

    H_inv = np.linalg.inv(H)
    transformed_points = np.dot(H_inv, Four_corners.T).T
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            orig_point = np.dot(H_inv, [j - shifty, i - shiftx, 1])
            orig_point = orig_point / orig_point[2]
            x, y = int(orig_point[0]), int(orig_point[1])
            if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
                canvas[i, j] = img1[y, x]

    # # Create a mask of the same size as the source image
    # mask = np.ones_like(img1, np.uint8)*255

    # # Find the center of the source image
    # center = (img1.shape[1]//2, img1.shape[0]//2)

    # # Use seamlessClone to blend the images
    # result = cv2.seamlessClone(img1.astype(np.uint8), canvas.astype(np.uint8), mask, center, cv2.NORMAL_CLONE)

    return canvas

def compute_homography(matched_keypoints):
    # Create a matrix to store the coefficients of the system of linear equations
    A = np.zeros((2 * len(matched_keypoints), 9))

    # For each pair of matched keypoints
    for i, (point1, point2) in enumerate(matched_keypoints):
        x1, y1, _ = point1[:,0]
        x2, y2, _ = point2[:,0]

        # Add the coefficients to the matrix

        A[(2 * i)] =  [-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2]

        A[((2 * i) + 1)] = [0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2]

    # Perform singular value decomposition
    _, _, V = np.linalg.svd(A)

    # The homography matrix is the last column of V
    H = V[-1].reshape((3, 3))

    return H




def main():
    # Load the images from the given folder and visualize the input images
	image_folder_name = "./P3Data/"
	op_folder_name = "./Results"
	if not os.path.exists(op_folder_name):
		os.makedirs(op_folder_name)	

	# FIle/Image handling functions call
	folder_names = os.listdir(image_folder_name)
    ipdb.set_trace()
	loaded_images = loadImages(image_folder_name, folder_names)

	create_visualization(loaded_images, folder_names, 3, (14, 5),op_folder_name+"/"+"Input.png")
	visualize= []
	Label = []  
	sequence_list = [[0,1,2,3,4]]	
	for sequence in sequence_list:
		new_image = None	
		for i in range(len(sequence)):
			# method = "Harris"   # Harris corner detection
			if i == len(loaded_images)-1:	
				break
			if new_image is None:
				first =loaded_images[sequence[i]]
				second = loaded_images[sequence[i+1]]	
			else:	
				first = loaded_images[sequence[i+1]]
				second = new_image	
			method = "good_features_to_track"  # Shi-Tomasi corner detection
			
			detected_corners1 , temp1 = detect_corners(first, method)
			detected_corners2 , temp2 = detect_corners(second, method)
			print("Corners detected in image ", i)
			# Perform ANMS on the detected corners
			anms_corners1 = anms(detected_corners1, method)
			anms_corners2 = anms(detected_corners2, method)	
			print("ANMS performed on the detected corners in image ", i)
			# Find feature descriptors for the detected corners
			descriptors1 , keypoints1 = find_feature_descriptors(first, anms_corners1)
			keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in list(keypoints1)]
			
			descriptors2 , keypoints2 = find_feature_descriptors(second, anms_corners2)
			keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in list(keypoints2)]	
					
			# Match the features between the two images
			matched_features = match_features(descriptors1, descriptors2)
			matched_image = cv2.drawMatches(first, keypoints1, second, keypoints2, matched_features, None, flags=2)
			visualize.append(matched_image) 
			Label.append("Matched features"+str(i))
			plt.imshow(matched_image)
			plt.show()
			# Perform RANSAC to find the best homography

			H_matrix, filtered_matches = ransac(keypoints1, keypoints2, matched_features, 1000, 10)
			if filtered_matches is None:
				continue
			# calculate homography from RANSAC matches using SVD
			H_matrix = compute_homography(filtered_matches)
			

		create_image(new_image,op_folder_name+"/"+"Case2_"+str(sequence)+".png")


if __name__ == "__main__":

    main()