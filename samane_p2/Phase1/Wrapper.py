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
from GetInliersRANSAC import ransac_wa, ransac_wh
from EstimateFundamentalMatrix import find_fundamental_mat, compute_fun_matrix
from EssentialMatrixFromFundamentalMatrix import find_essential_mat
from ExtractCameraPose import find_camera_pose
from LinearTriangulation import find_initial_traingulation

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

def store_match_features(txt_file):
    image_dict_list = []


    with open(txt_file, 'r') as file:
        lines = file.readlines()
        nFeatures = int(lines[0].split()[1])
        for i in range(1,len(lines)):
            temp=0
            for j in range(int(lines[i].split()[0])-1):
                image_dict={}
                image_dict["ref"] = int(lines[i].split()[6+temp])
                image_dict["rgd"] = (int(lines[i].split()[1]),int(lines[i].split()[2]),int(lines[i].split()[3]))
                image_dict["keypoint1"] = cv2.KeyPoint(float(lines[i].split()[4]), float(lines[i].split()[5]), 1)
                image_dict["keypoint2"] = cv2.KeyPoint(float(lines[i].split()[7+temp]), float(lines[i].split()[8+temp]), 1)
                image_dict_list.append(image_dict)
                temp += 3

    return image_dict_list
     
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

def draw_epipolar_lines(F,final_matching_pairs,image):
    x= np.linspace(0,image.shape[1])
    for i, [point1, point2] in enumerate(final_matching_pairs):
        ul, vl = point1
        y = (-(F[0,2]*ul + F[1,2]*vl+F[2,2]) - (F[0,0]*ul + F[1,0]*vl + F[2,0])* x)/(F[0,1]*ul + F[1,1]*vl + F[2,1])
        plt.plot(x,y)
    
    plt.imshow(image)
    plt.show()

def normalize_image_coordinates(final_matching_pairs,image1,image2):
    max_y, max_x, _ = image1.shape
    for i in range(len(final_matching_pairs)):
        final_matching_pairs[i][0][0] = final_matching_pairs[i][0][0] / max_x
        final_matching_pairs[i][0][1] = final_matching_pairs[i][0][1] / max_y

        final_matching_pairs[i][1][0] = final_matching_pairs[i][0][0] / max_x
        final_matching_pairs[i][1][1] = final_matching_pairs[i][0][1] / max_y
    return final_matching_pairs



     
    

def main():
    input_folder_name = "./P3Data/"
    op_folder_name = "./Results"
    if not os.path.exists(op_folder_name):
        os.makedirs(op_folder_name)	
    folder_names = os.listdir(input_folder_name)
    image_files = []
    match_text_file=[]
    matched_features_list = []
    num_images = 5
    
    for file in folder_names:
        if ".png" in file:
            image_files.append(file)
        elif "matching" in file:
            match_text_file.append(file)
            matched_features_list.append(store_match_features(input_folder_name + file))
            print("txt file appended",file)

        elif "calibration" in file:
            calib_file_path = input_folder_name + file
            with open(calib_file_path, 'r') as file:
                lines = file.readlines()
                calib_mat = []
                for line in lines:
                    calib_mat.append(list(map(float, line.split())))
            calib_mat = np.matrix(calib_mat)
            
    images = loadImages(input_folder_name,image_files)
    keypoints1=[]
    keypoints2=[]
    points1 = []
    points2 = []
    matches = []
    x=0

    for i in range(len(matched_features_list[0])):
        
        if matched_features_list[0][i]["ref"]==2:
            keypoints1.append(matched_features_list[0][i]["keypoint1"])
            points1.append([matched_features_list[0][i]["keypoint1"].pt[0],matched_features_list[0][i]["keypoint1"].pt[1]])
            keypoints2.append(matched_features_list[0][i]["keypoint2"])
            points2.append([matched_features_list[0][i]["keypoint2"].pt[0],matched_features_list[0][i]["keypoint2"].pt[1]])

            matches.append(cv2.DMatch(x, x, 0))
            x+=1
    
    # For image 1 and image 2
    visualize=[]
    label = []
    matched_image = cv2.drawMatches(images[0], keypoints1, images[1], keypoints2, matches, None, flags=2)

    # plt.imshow(matched_image)
    # plt.show()
    visualize.append(matched_image)
    label.append("Before Ransac")
    H_matrix, filtered_matches, final_matching_pairs = ransac_wh(keypoints1, keypoints2, matches, 1000, 10)
    A_matrix, filtered_matches, final_matching_pairs = ransac_wa(keypoints1, keypoints2, filtered_matches, 1000, 0.05)
    # calculate homography from RANSAC matches using SVD
    matched_image = cv2.drawMatches(images[0], keypoints1, images[1], keypoints2, filtered_matches, None, flags=2)
    # plt.imshow(matched_image)
    # plt.show()
    visualize.append(matched_image)
    label.append("After Ransac")
    create_visualization(visualize,label,1,(14, 7),op_folder_name+"/"+"ransac.png")
    norm_matching_pairs = normalize_image_coordinates(final_matching_pairs,images[0],images[1])
    Final_F_matrix = compute_fun_matrix(norm_matching_pairs)
    draw_epipolar_lines(Final_F_matrix,final_matching_pairs,images[1] )
    Ess_matrix = find_essential_mat(Final_F_matrix, calib_mat)
    
    camera_poses = find_camera_pose(Ess_matrix)
    find_initial_traingulation(camera_poses, norm_matching_pairs, calib_mat)





              

        



if __name__ == "__main__":
    main()