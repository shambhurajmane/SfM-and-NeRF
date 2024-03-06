
import os 
import numpy as np
import json
import torch
import cv2
import math
from functools import reduce
import ipdb
import matplotlib.pyplot as plt 

class NeRFDataSetLoader():
    def __init__(self, model_type, root_dir):
        self.root_dir = root_dir
        
        #self.tiny_data = np.load("tiny_nerf_data.npz"):
        if model_type == "train":
            self.data_file = "transforms_train.json"
        elif model_type == "test":
            self.data_file = "transforms_test.json"
        elif model_type == "val":
            self.data_file = "transforms_val.json"
        else:
            pass 
        file_path = root_dir + self.data_file

        #open file
        with open(file_path) as file:
            self.data = json.load(file)
        
        #print(type(self.data))
        #print(self.data["frames"][idx]["transform_matrix"])

    def center_crop(image, crop_size):
        """
        Center crop the input image to the specified size.

        Parameters:
        - image: Input image (NumPy array).
        - crop_size: Tuple (width, height) specifying the target crop size.

        Returns:
        - Cropped image.
        """

        # Get image dimensions
        height, width = image.shape[:2]

        # Calculate crop coordinates
        crop_x = max(0, (width - crop_size[0]) // 2)
        crop_y = max(0, (height - crop_size[1]) // 2)

        # Perform the crop
        cropped_image = image[crop_y:crop_y + crop_size[1], crop_x:crop_x + crop_size[0]]

        return cropped_image
    
    
    def getitem(self):
        
        image_list = [] 
        camera_pose_list = []
        camera_info_list = []
        #image_path = os.path.join(self.root_dir +os.sep + image_name)
        for idx in range(0,len(self.data["frames"])):
            path = self.data["frames"][idx]["file_path"]
            path = path.split("./")[-1]
            path = self.root_dir + path+".png"
            #print(path)
            image  = cv2.imread(path)             
            #print(type(image))
            # image = cv2.resize(image, (50,50))
            crop_size = (500,500)
            image = NeRFDataSetLoader.center_crop(image, crop_size)
            image = cv2.resize(image, (50,50))
            
    
            image_list.append( torch.tensor(image))
            camera_pose_list.append( torch.tensor(self.data["frames"][idx]["transform_matrix"]))
            camera_angle_x =self.data["camera_angle_x"]
            height, width = image.shape[0], image.shape[1]
            focal_length = 0.5* image.shape[0]/math.tan(0.5* camera_angle_x)
            camera_matrix = np.array([[focal_length, 0, width/2], [0, focal_length, height/2], [0, 0, 1]])
            camera_info_list.append([width, height, torch.tensor(camera_matrix)])
        image_tensor = torch.stack(image_list )    
        camera_pose_tensor =  torch.stack(camera_pose_list )
        camera_info_tensor = camera_info_list
        
        
    
        # Wait for a key press and then close the window
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(image.shape)
        #focal_length = 0.5* image.shape[0]/math.tan(0.5* camera_angle_x) 
        return image_tensor, camera_pose_tensor, camera_info_tensor

    def get_mini_batch(input_data, batch_size):
        index = range(0, input_data.shape[0], batch_size)
        batch = input_data[index]