import torch
import math
import numpy as np
def ndc_rays(height, width, focal_length, near,  ray_origin, rays_direction):
    "shift ray origin to a place nearby plane"
    t = -(near +  ray_origin[..., 2]) / rays_direction[..., 2]
    ray_origin = ray_origin + t[..., None] * rays_direction

    o0 = -1./(width/(2.*focal_length)) * ray_origin[..., 0] / ray_origin[..., 2]
    o1 = -1./(height/(2.*focal_length)) * ray_origin[..., 1] / ray_origin[..., 2]
    o2 = 1. + 2. * near / ray_origin[..., 2]

    d0 = -1./(width/(2.*focal_length)) * \
        (rays_direction[..., 0]/rays_direction[..., 2] - ray_origin[..., 0]/ray_origin[..., 2])
    
    d1 = -1./(height/(2.*focal_length)) * \
        (rays_direction[..., 1]/rays_direction[..., 2] - ray_origin[..., 1]/ray_origin[..., 2])
    d2 = -2. * near / ray_origin[..., 2]

    ray_origin = tf.stack([o0, o1, o2], -1)
    rays_direction = tf.stack([d0, d1, d2], -1)

    return ray_origin, rays_direction


def get_rays(height, width, focal_length, camera_pose):
    """
    height: height of the imgae
    width: width of the image
    focal_length : focal_length length of the camera
    camera_pose : The projecction matrix of the camera, it is a transoformation matrix.  

    Output:
    Ray traced throguth 3D volume of the scene
    """
    rotation_mat = camera_pose[1:3, 1:3]
    i, j  = torch.meshgrid(height, width) 
    m = i.transpose(-1,-2)
    n = j.transpose(-1,-2)

    x = (i-width* 0.5)/focal_length
    y = -(j-height * 0.5)/focal_length

    ray = torch.stack([x, y, -torch.ones_like(i)],dim = -1)
    directions = ray[..., np.newaxis, :]
    ray_direction = torch.sum( directions * rotation_mat, dim= -1)
    ray_origin = camera_pose[:3,-1].expand(ray_direction.shape)

    return ray_origin,ray_direction

def positional_encoding(points, max_frequency, no_term):
    "This applies positional encoding on"
    
    encoded_points = [points] 
    freq_bands = torch.linspace( 1 , 2.**max_frequency , no_term)
    for freq in freq_bands:
        encoded_points.append(torch.sin( freq* points))
        encoded_points.append(torch.cos( freq * points))
    
    return encoded_points


        

    


if __name__ == "__main__":
    positional_encoding(1,2)