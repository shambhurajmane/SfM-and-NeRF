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
from BundleAdjustment import *  

op_folder_name = "./Full_Results"
if not os.path.exists(op_folder_name):
    os.makedirs(op_folder_name)

def create_visualization(images, Labels, cols, size,file_name):
	rows = int(np.ceil(len(images)/cols))
	plt.subplots(rows, cols, figsize=size)
	for index in range(len(images)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(images[index], cmap= "gray")
		plt.title(Labels[index])
	plt.savefig(file_name)	
	# plt.show()
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
        left_ref = int(txt_file.split("/")[2][8])
        lines = file.readlines()
        nFeatures = int(lines[0].split()[1])
        for i in range(1,len(lines)):
            temp=0
            for j in range(int(lines[i].split()[0])-1):
                image_dict={}
                image_dict["left_ref"] = left_ref
                image_dict["right_ref"] = int(lines[i].split()[6+temp])
                
                image_dict["rgd"] = (int(lines[i].split()[1]),int(lines[i].split()[2]),int(lines[i].split()[3]))
                image_dict["keypoint1"] = [float(lines[i].split()[4]), float(lines[i].split()[5])]
                image_dict["keypoint2"] = [float(lines[i].split()[7+temp]), float(lines[i].split()[8+temp])]
                image_dict_list.append(image_dict)
                temp += 3

    return image_dict_list  

def get_matching_points(ref_image_id,matching_image_id,matched_features_list):
    matched_points = []
    for i in range(len(matched_features_list[0])):
    
        if matched_features_list[0][i]["left_ref"]==ref_image_id and matched_features_list[0][i]["right_ref"]==matching_image_id:
            matched_points.append([matched_features_list[0][i]["keypoint1"],matched_features_list[0][i]["keypoint2"]])
    return  matched_points

def draw_epipolar_lines(F,norm_matching_pairs,image,title):
    x= np.linspace(0,image.shape[1])
    for i, [point1, point2] in enumerate(norm_matching_pairs):
        ul, vl = point1

        y = (-(F[0,2]*ul + F[1,2]*vl+F[2,2]) - (F[0,0]*ul + F[1,0]*vl + F[2,0])* x)/(F[0,1]*ul + F[1,1]*vl + F[2,1])
        plt.plot(x,y)
    plt.title(title)
    plt.imshow(image)
    plt.savefig(op_folder_name + "/" + title + ".png")
    # plt.show()
    plt.close()

def check_reprojection(filtered_matched_points,world_points,projections, image_left, image_right, title):

    plt.subplots(1, 2, figsize=(8, 14))
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
        plt.subplot(2, 1, 1)
        plt.imshow(image_left)
        plt.scatter(left_pt[0], left_pt[1], color='red',s=5)
        plt.scatter(left_proj[0], left_proj[1], color='blue',s=5)
        plt.title(title + "left image")


        plt.subplot(2, 1, 2)
        plt.imshow(image_right)
        plt.scatter(right_pt[0], right_pt[1], color='red', s=5)
        plt.scatter(right_proj[0], right_proj[1], color='blue', s=5)
        plt.title(title + "right image")

        

    plt.legend(['Projections', 'Reprojection'])
    plt.savefig(op_folder_name + "/" + title + ".png")
    # plt.show()
    plt.close()


def visualize_points_camera_poses(visualize, camera_pose, title):
    colors = ['r', 'b', 'g', 'y', 'c', 'm']
    for i in range(len(visualize)):
        points_list = np.asarray(visualize[i])              

        x_pts_1, y_pts_1, z_pts_1 = points_list[:, 0], points_list[:, 1], points_list[:, 2]

        dot_size = 1
        axes_lim = 20

        plt.scatter(x_pts_1, z_pts_1, color=colors[i], s=dot_size)

    for i in range(len(camera_pose)):

        r2 = Rotation.from_matrix(camera_pose[i][1])
        angles2 = r2.as_euler("zyx", degrees=True)

        plt.plot(camera_pose[i][0][0], camera_pose[i][0][2], marker=(3, 0, int(angles2[1])), markersize=15, linestyle='None', color=colors[i])

    plt.title("triangulated world points")
    plt.xlim(-axes_lim, axes_lim)
    plt.ylim(-5, 30)
    plt.xlabel("x (dimensionless)")
    plt.ylabel("z (dimensionless)")
    plt.legend(['Before optimization', 'After optimization'])
    plt.savefig(op_folder_name + "/" + title + ".png")

    # show plot
    # plt.show()
    plt.close()

def find_matches( world_points_corrected , matches_1_2 , matches_1_3 , u_v_1_3 ):
    """
    Use matches between 1-3 and 1-2, to find corresponding matching between img 2 and 3 related through 1. 
    """
    world_points_corrected = np.asarray(world_points_corrected)
    u_v_1_12 = matches_1_2
    u_v_1_13 = matches_1_3
    
    mask = np.isin(u_v_1_12, u_v_1_13)
    indices_1 = np.where(mask == 1)[0] 
    indices_1 = np.unique(indices_1) 
    u_v_12 = u_v_1_12[indices_1] 

    mask_2 = np.isin(u_v_1_13, u_v_12 )
    indices_2 = np.where(mask_2 == 1)[0] 
    indices_2 = np.unique(indices_2) 
    u_v_13 = u_v_1_13[indices_2]

    world_points_13 = world_points_corrected[indices_1]
    u_v_12 = matches_1_2[indices_1]

    return u_v_13, world_points_13, u_v_12


def correspondences_2D_to_3D(world_points_12, Feature_12, Feature_1i,image1, image2, index):

    world_points_1i = []
    x2D_in_id = []
    x2D_in_1 = []
    matches_1i = []
    # visualize_matches(Feature_1i, image1, image2) 

    F1_from_12= Feature_12[:,0]
    F1_from_1i= Feature_1i[:,0]
    Fi_from_1i= Feature_1i[:,1]
    
    for i in range(len(F1_from_12)):
        for j in range(len(F1_from_1i)):
            condn1 = abs(int(F1_from_1i[j][0]) - int(F1_from_12[i][0]))<2
            condn2 = abs(int(F1_from_1i[j][1]) - int(F1_from_12[i][1])) <2
        

            if condn1 and condn2:
                world_points_1i.append(world_points_12[i])
                x2D_in_1.append(F1_from_12[i])
                x2D_in_id.append(Fi_from_1i[j])
                matches_1i.append([F1_from_12[i], Fi_from_1i[j]])
                break
    print("num matched points: ", len(matches_1i))  
    visualize_matches(matches_1i, image1, image2, title= "Matches between 1 and {}".format(index)  )

    return np.array(x2D_in_1),np.array(x2D_in_id), np.array(world_points_1i)

def get_camera_params(camera_poses, k):
    # Get 9 camera parameters from camera poses and intrinsic matrix which are used in bundle adjustment
    # 9 camera parameters are 3 rotation angles, 3 translation values, 1 focal length and 2 distortion coefficients
    camera_params = []
    for i in range(len(camera_poses)):
        R = camera_poses[i][1]
        t = camera_poses[i][0]
        rvec, _ = cv2.Rodrigues(R)
        camera_param = np.append(rvec, t)
        camera_param = np.append(camera_param, k[0, 0])
        camera_param = np.append(camera_param, k[0, 2])
        camera_param = np.append(camera_param, k[1, 2])

        camera_params.append(camera_param)  
    return np.array(camera_params)
    
def projectpoints(points, image):
    image_left = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    

    for i in range(len(points)):
        left_pt = points[i][0]
        plt.imshow(image_left)
        plt.scatter(left_pt[0], left_pt[1], color='red',s=5)
    plt.imsave(op_folder_name + "/projected_points.png", image_left)
    # plt.show() 
    plt.close()



def visualize_matches( matched_points,image1, image2, title):
    """
    Mark out and draw each matched corner pair on the two images
    :param num_image_1: first image number
    :param num_image_2: second image number
    :param matched_points: The array of matched corner points of shape [n x 2 x 2]
    """
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)    
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    if len(image1.shape) == 3:
        height1, width1, depth1 = image1.shape
        height2, width2, depth2 = image2.shape
        shape = (max(height1, height2), width1 + width2, depth1)

    elif len(image1.shape) == 2:
        height1, width1, depth1 = image1.shape
        height2, width2, depth2 = image2.shape
        shape = (max(height1, height2), width1 + width2)

    image_combined = np.zeros(shape, type(image1.flat[0]))  
    image_combined[0:height1, 0:width1] = image1 
    image_combined[0:height1, width1:width1 + width2] = image2 
    image_1_2 = image_combined.copy()

    circle_size = 4
    red = [255, 0, 0]
    cyan = [0, 255, 255]
    yellow = [255, 255, 0]

    for i in range(len(matched_points)):
        corner1_x = matched_points[i][0][0]
        corner1_y = matched_points[i][0][1]
        corner2_x = matched_points[i][1][0]
        corner2_y = matched_points[i][1][1]

        cv2.line(image_1_2, (int(corner1_x), int(corner1_y)), (int(corner2_x + image1.shape[1]), int(corner2_y)), red,
                 1)
        cv2.circle(image_1_2, (int(corner1_x), int(corner1_y)), circle_size, cyan, 1)
        cv2.circle(image_1_2, (int(corner2_x) + image1.shape[1], int(corner2_y)), circle_size, yellow, 1)

    scale = 1.5
    height = image_1_2.shape[0] / scale
    width = image_1_2.shape[1] / scale
    im = cv2.resize(image_1_2, (int(width), int(height)))

    plt.imshow(im)
    plt.title(title)
    plt.imsave(op_folder_name + "/" + title + ".png", im)
    plt.close()



def main():
    """

    """
    input_folder_name = "./P3Data/"
    op_folder_name = "./Full_Results"
    if not os.path.exists(op_folder_name):
        os.makedirs(op_folder_name)	
    folder_names = os.listdir(input_folder_name)
    image_files = []
    match_text_file=[]
    matched_features_list = []
    num_images = 5
    vis  = False

    
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
    K = calib_mat


    # Matching features between images 1 and 2
    ref_image_id = 1
    matching_image_id = 2
    matched_points = np.asarray(get_matching_points(ref_image_id,matching_image_id,matched_features_list))   
    if vis:
        visualize_matches(matched_points, images[ref_image_id-1], images[matching_image_id-1], title= "Initial matches between 1 and 2")  
    
    # Matching features between images 1 and 2 performed using RANSAC and computing the fundamental matrix
    Final_fundamental_mat, filtered_match_pts_12 = get_inliers_RANSAC(matched_points,iterations = 1000, threshold = 0.05)
    if vis:
        visualize_matches(filtered_match_pts_12, images[ref_image_id-1], images[matching_image_id-1], title= "RANSAC matches between 1 and 2") 
        draw_epipolar_lines(Final_fundamental_mat,filtered_match_pts_12,images[1], title= "Epipolar lines between 1 and 2")

    
    # Extract essential matrix from fundamental matrix
    Essential_mat = find_essential_mat(Final_fundamental_mat, K)

    #Extract camera poses from Essential matrix
    camera_poses = extract_camera_pose(Essential_mat)

    # Perform linear triangulation to disambiguate camera poses
    Xr_points = []
    projections = []
    for camera_pose in camera_poses:                
        X_r, pr_r, pr_l = find_linear_traingulation(K, camera_pose, filtered_match_pts_12)
        Xr_points.append(X_r)
        projections.append([pr_r,pr_l])

    correct_pose, world_points, index = disambiguate_camera_poses(camera_poses, Xr_points)
    if vis:
        check_reprojection(filtered_match_pts_12,world_points,projections[index], images[0], images[1], title= "Triangulation before optimizaztion in ")
        visualize_points_camera_poses([world_points], [correct_pose], title= "Camera poses and 3D points depth before optimization between 1 and 2")

    # Perform non-linear triangulation to optimize the 3D points
    world_points_corrected = non_linear_triangulation( filtered_match_pts_12, world_points,projections[index])
    if vis:
        check_reprojection(filtered_match_pts_12, world_points_corrected,projections[index], images[0], images[1], title= "Triangulation  After optimizaztion in ")
    final_camera_poses = []
    final_camera_poses.append([np.zeros(3), np.eye(3)])     # Camera pose for visibilty matrix of points in camera 1
    final_camera_poses.append(correct_pose)                 # Camera pose for visibilty matrix of points in camera 2
    visualize = [world_points,world_points_corrected]
    if vis:
        visualize_points_camera_poses(visualize , final_camera_poses, title = "Camera poses and 3D points depth after optimization between 1 and 2")
    
    
    # Loop over the remaining images 2 - 5
    depth_points = []
    depth_points.append(world_points_corrected) # 3D points for visibilty matrix of points in camera 1
    depth_points.append(world_points_corrected) # 3D points for visibilty matrix of points in camera 2

    for image_num in range(3, 6):

        ref_image_id = 1
        matching_image_id = image_num

        # Matching features between images 1 and image_num
        matched_points = np.asarray(get_matching_points(ref_image_id,matching_image_id,matched_features_list))  

        # Matching features between images 1 and image_num performed using RANSAC and computing the fundamental matrix
        F, filtered_mp = get_inliers_RANSAC(matched_points,iterations = 3000, threshold = 0.01)
        if vis:
            draw_epipolar_lines(Final_fundamental_mat,filtered_mp,images[image_num-1], title= "Epipolar lines between 1 and {}".format(image_num))
            visualize_matches(filtered_mp, images[ref_image_id-1], images[matching_image_id-1], title= "RANSAC matches between 1 and {}".format(image_num))

        # Find the matching features between images 1 and image_num using the matches between 1 and 2
        x2D_in_1,x2D_in_i, X3D_in_i = correspondences_2D_to_3D(world_points_corrected, filtered_match_pts_12, filtered_mp, images[ref_image_id-1], images[matching_image_id-1], image_num )
        
        # Perform PnP RANSAC to find the camera pose
        camera_pose_new, prr = PnP_RANSAC(K, x2D_in_i, X3D_in_i,iterations = 5000,threshold = 5)
        C_new, R_new = camera_pose_new
        
        matches = []
        for ss in range(len(x2D_in_i)):
             matches.append([x2D_in_1[ss],x2D_in_i[ss]])
        matches = np.array(matches)
        if vis:
            check_reprojection(matches,X3D_in_i,prr, images[0], images[image_num-1],title= "pnpRANSAC_before_optimization with image {}".format(image_num))

        # Perform non-linear PnP to optimize the camera pose and 3D points
        C_opt, R_opt, pr_r, pr_l = nonlinear_PnP(K, x2D_in_i, X3D_in_i, R_new, C_new)

        # Append the camera pose and 3D points to the list
        final_camera_poses.append([C_opt, R_opt])   # Camera pose for visibilty matrix of points in camera 3,4,5
        # X_r, pr_r, pr_l  = find_linear_traingulation(K, [C_opt, R_opt], matches)

        # X_points_nonlin = non_linear_triangulation( matches, X_r, [pr_r, pr_l])
        if vis:
            check_reprojection(matches,X3D_in_i,[pr_r, pr_l], images[0], images[image_num-1], title= "pnpRANSAC_after_optimization with image {}".format(image_num))



        # depth_points.append(X3D_in_i)        # 3D points for visibilty matrix of points in camera 3,4,5
    
    # Visualize all camera poses and 3D points before bundle adjustment
    visualize_points_camera_poses(depth_points, final_camera_poses, title = "Camera poses and 3D points depth before bundle adjustment for all images")

    # Build Visibility matrix and perform bundle adjustment
    camera_params = get_camera_params(final_camera_poses, K) 
    n_cameras, n_points, camera_indices, point_indices, points_2d,points_3d = build_visibility_matrix(final_camera_poses, K, depth_points, images)
    final_camera_poses, world_pts = bundle_adjustment(n_cameras, n_points, camera_params, points_3d, camera_indices, point_indices, points_2d)

     # Visualize all camera poses and 3D points before bundle adjustment
    visualize_points_camera_poses([world_pts], final_camera_poses, title = "Camera poses and 3D points depth after optimization for all images")


if __name__ == "__main__":
    main()