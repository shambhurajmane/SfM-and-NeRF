"""
RBE/CS Spring 2023: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 2

"""

import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation
from GetInliersRANSAC import *
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import * 
from ExtractCameraPose import *
from LinearTriangulation import *
from PnPRANSAC import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *


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
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                image_dict["keypoint1"] = [float(lines[i].split()[4]), float(lines[i].split()[5])]
                image_dict["keypoint2"] = [float(lines[i].split()[7+temp]), float(lines[i].split()[8+temp])]
                image_dict_list.append(image_dict)
                temp += 3

    return image_dict_list  

def get_matching_points(ref_image_id,matching_image_id,matched_features_list):
    matched_points = []
    for i in range(len(matched_features_list[0])):
    
        if matched_features_list[0][i]["ref"]==matching_image_id:
            matched_points.append([matched_features_list[0][i]["keypoint1"],matched_features_list[0][i]["keypoint2"]])
    return  matched_points

def draw_epipolar_lines(F,norm_matching_pairs,image):
    x= np.linspace(0,image.shape[1])
    for i, [point1, point2] in enumerate(norm_matching_pairs):
        ul, vl = point1

        y = (-(F[0,2]*ul + F[1,2]*vl+F[2,2]) - (F[0,0]*ul + F[1,0]*vl + F[2,0])* x)/(F[0,1]*ul + F[1,1]*vl + F[2,1])
        plt.plot(x,y)
    
    plt.imshow(image)
    plt.show()

def check_reprojection(filtered_matched_points,world_points,projections, image_left, image_right, title):

    plt.subplots(1, 2, figsize=(14, 5))
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)    
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)

    for i in range(len(world_points)):
        left_pt = filtered_matched_points[i][0]
        right_pt = filtered_matched_points[i][1]  
        ur, vr, wr = world_points[i]
        wp = np.array([ur, vr, wr,1])
        left_proj = np.matmul(projections[0],wp)
        left_proj = left_proj / left_proj[-1]

        right_proj = np.matmul(projections[1],wp)
        right_proj = right_proj / right_proj[-1]
        plt.subplot(1, 2, 1)
        plt.imshow(image_left)
        plt.scatter(left_pt[0], left_pt[1], color='red',s=5)
        plt.scatter(left_proj[0], left_proj[1], color='blue',s=5)
        plt.title(title + "image1")


        plt.subplot(1, 2, 2)
        plt.imshow(image_right)
        plt.scatter(right_pt[0], right_pt[1], color='red', s=5)
        plt.scatter(right_proj[0], right_proj[1], color='blue', s=5)
        plt.title(title + "image2")

        

    plt.legend(['Projections', 'Reprojection'])
    plt.show()


def visualize_points_camera_poses(points_list, currected_points_list, camera_pose):

    points_list = np.asarray(points_list)              

    x_pts_1, y_pts_1, z_pts_1 = points_list[:, 0], points_list[:, 1], points_list[:, 2]

    dot_size = 1
    axes_lim = 20

    plt.scatter(x_pts_1, z_pts_1, color="red", s=dot_size)

    points_list = np.asarray(currected_points_list)              

    x_pts_1, y_pts_1, z_pts_1 = points_list[:, 0], points_list[:, 1], points_list[:, 2]

    dot_size = 1
    axes_lim = 20

    plt.scatter(x_pts_1, z_pts_1, color="blue", s=dot_size)
    # plt.scatter(x_pts_2, z_pts_2, color="blue", s=dot_size)

    for i in range(len(camera_pose)):

        r2 = Rotation.from_matrix(camera_pose[i][1])
        angles2 = r2.as_euler("zyx", degrees=True)

        plt.plot(camera_pose[i][0][0], camera_pose[i][0][2], marker=(3, 0, int(angles2[1])), markersize=15, linestyle='None')

    plt.title("triangulated world points")
    plt.xlim(-axes_lim, axes_lim)
    plt.ylim(-5, 30)
    plt.xlabel("x (dimensionless)")
    plt.ylabel("z (dimensionless)")
    plt.legend(['Before optimization', 'After optimization'])

    # show plot
    plt.show()


def main():
    """

    """
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
            calib_mat = np.array(calib_mat)
            
    images = loadImages(input_folder_name,image_files)
    ref_image_id = 1
    matching_image_id = 2

    matched_points = np.asarray(get_matching_points(ref_image_id,matching_image_id,matched_features_list))     
    
    print('num matched points: ', len(matched_points))

    Final_fundamental_mat, filtered_matched_points = get_inliers_RANSAC(matched_points,iterations = 1000, threshold = 0.05)
    print("F: ", Final_fundamental_mat)
    print('num best matched points: ', len(filtered_matched_points))

    draw_epipolar_lines(Final_fundamental_mat,filtered_matched_points,images[1])

    K = calib_mat

    Essential_mat = find_essential_mat(Final_fundamental_mat, K)
    print('E: ', Essential_mat)

    #Extract camera poses from Essential matrix
    camera_poses = extract_camera_pose(Essential_mat)
    # C_list, R_list = extract_camera_pose(Essential_mat)

    Xr_points = []
    projections = []

    for camera_pose in camera_poses:                
        X_r, pr_r, pr_l = find_linear_traingulation(K, camera_pose, filtered_matched_points)
        Xr_points.append(X_r)
        projections.append([pr_r,pr_l])

    correct_pose, world_points, index = disambiguate_camera_poses(camera_poses, Xr_points)
    check_reprojection(filtered_matched_points,world_points,projections[index], images[0], images[1], title= "Before optimizaztion")
    final_camera_poses = []
    final_camera_poses.append([np.zeros(3), np.eye(3)])
    final_camera_poses.append(correct_pose)
    # visualize_points_camera_poses(world_points, final_camera_poses)

    world_points_corrected = non_linear_triangulation( filtered_matched_points, world_points,projections[index])
    check_reprojection(filtered_matched_points, world_points_corrected,projections[index], images[0], images[1], title= "After optimizaztion")
    visualize_points_camera_poses(world_points,world_points_corrected , final_camera_poses)


     




if __name__ == "__main__":
    main()