
import os 
import numpy as np
import json
import torch
import cv2
import math
from functools import reduce

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

    def getitem(self, idx):
        if torch.is_tensor(idx):
            idx_list = idx.tolist()

        #image_name = [self.root_dir + self.data["frames"][idx]["file_path"]+".png" for idx in  idx_list ]
        #camera_pose = [self.data["frames"][idx]["transform_matrix"] for idx in idx_list]
        #print(image_name)
        image_list = [] 
        camera_pose_list = []
        camera_info_list = []
        #image_path = os.path.join(self.root_dir +os.sep + image_name)
        for idx in idx_list:
            path = self.data["frames"][idx]["file_path"]
            path = path.split("./")[-1]
            path = self.root_dir + path+".png"
            #print(path)
            image  = cv2.imread(path)             
            #print(type(image))
            image = cv2.resize(image, (250,250))
            image_list.append( torch.tensor(image))
            camera_pose_list.append( torch.tensor(self.data["frames"][idx]["transform_matrix"]))
            camera_angle_x =self.data["camera_angle_x"]
            height, width = image.shape[0], image.shape[1]
            focal_length = 0.5* image.shape[0]/math.tan(0.5* camera_angle_x)
            camera_matrix = np.array([[focal_length, 0, width/2], [0, focal_length, height/2], [0, 0, 1]])
            camera_info = [width, height, camera_matrix]
            camera_info_list.append(torch.tensor(camera_info))
        image_tensor = torch.stack(image_list )    
        camera_pose_tensor =  torch.stack(camera_pose_list )
        camera_info_tensor = torch.tensor(camera_info_list) 
        
        
    
        # Wait for a key press and then close the window
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #print(image.shape)
        #focal_length = 0.5* image.shape[0]/math.tan(0.5* camera_angle_x) 
        return image_tensor, camera_pose_tensor, camera_info_tensor

    def get_mini_batch(input_data, batch_size):
        index = range(0, input_data.shape[0], batch_size)
        batch = input_data[index]