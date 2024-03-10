import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import wandb
from NeRFModel import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import ipdb 
from Utils import NeRFDataSetLoader
np.random.seed(0)
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """
    loader = NeRFDataSetLoader(mode, data_path)
    images, poses, camera_info = loader.getitem()
    #print(images.shape)
    #print(poses.shape)
    #print(camera_info.shape)
    #images, poses, camera_info = 1, 1, 1

    return images, poses, camera_info 


def PixelToRay(camera_info, pose, pixelPosition, args):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """
    # pixel position to ray origin and direction
    height, width, camera_matrix = camera_info
    focal_length = camera_matrix[0][0]
    camera_orgin = np.array([0, 0, 0])
    ray_origin_in_camera_coordinate = [0, 0, -1]
    pixel_x = (pixelPosition[0] - width / 2) / focal_length
    pixel_y = (pixelPosition[1] - height / 2) / focal_length
    pixel_position_in_camera_coordinate = np.array([pixel_x, pixel_y, 0])
    ray_direction_in_camera_coordinate = pixel_position_in_camera_coordinate - camera_orgin
    ray_direction_in_world_coordinate = np.dot(ray_direction_in_camera_coordinate, pose[:3, :3].T)
    ray_origin_in_world_coordinate = torch.add(torch.tensor(np.dot(ray_origin_in_camera_coordinate, pose[:3, :3].T)) ,pose[:3, 3])  #
    ray_origin = torch.tensor(ray_origin_in_world_coordinate)
    ray_direction = torch.tensor(ray_direction_in_world_coordinate)
    return ray_origin, ray_direction
    
def sampleRay(ray_origin, ray_direction, args):
    """
    Input:
        ray_origin: origins of input rays
        ray_direction: direction of input rays
        args: get sample rate
    Outputs:
        A set of rays
    """
    # sample rays
    delta = [0]
    t_vals = torch.linspace(args.near, args.far, args.n_sample)
    noise = torch.rand(t_vals.shape) * (args.far - args.near) / args.n_sample
    t_vals = t_vals + noise
    for i in range(0,t_vals.shape[0]-1):
        delta.append(t_vals[i+1] - t_vals[i])
        
    ray = ray_origin + t_vals[:, None] * ray_direction
    return ray , torch.tensor(delta)

def generateBatch(images, poses, camera_info,mode,count, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """
    
    # randomly select a set of rays
    image_index = random.randint(0, images.shape[0]-1)
    camera_pose = poses[image_index]
    camera_info = camera_info[image_index]
    image = images[image_index]
    batch_rays_origin = np.empty((args.n_rays_batch, args.n_sample, 3))
    batch_rays_direction = np.empty((args.n_rays_batch, args.n_sample, 3))
    batch_delta = np.empty((args.n_rays_batch, args.n_sample))
     
    batch_pixelColors = np.empty((args.n_rays_batch, 3))
    if mode == 'train':
        for i in range(args.n_rays_batch):
            # randomly select a pixel
            if count <args.precrop_iters:
                height, width = image.shape[:2]
                crop_size = (150, 150)  
                crop_x = max(0, (width - crop_size[0]) // 2)
                crop_y = max(0, (height - crop_size[1]) // 2)
                pixelPosition = random.randint(crop_y, crop_y + crop_size[1]-1), random.randint(crop_x,crop_x + crop_size[0]-1)
            else:
                pixelPosition = random.randint(0, camera_info[0]-1), random.randint(0, camera_info[1]-1)
            ray_origin, ray_direction = PixelToRay(camera_info, camera_pose, pixelPosition, args)
            ray_origins, delta = sampleRay(ray_origin, ray_direction, args)
            
            ray_directions = np.tile(ray_direction, (args.n_sample, 1))
            
            batch_rays_origin[i] = ray_origins
            batch_rays_direction[i] = ray_directions
            batch_delta[i] = delta

            color = image[pixelPosition[0], pixelPosition[1]]
            normalized_color = color / 255
            batch_pixelColors[i] = normalized_color
    elif mode == 'test':
        batch_rays_origin = np.empty((camera_info[0], camera_info[1], args.n_sample, 3))
        batch_rays_direction = np.empty((camera_info[0], camera_info[1], args.n_sample, 3))
        batch_delta = np.empty((camera_info[0], camera_info[1], args.n_sample))
        batch_pixelColors = np.empty((camera_info[0], camera_info[1], 3))

        for i in range(camera_info[0]):
            for j in range(camera_info[1]):
                # randomly select a pixel
                pixelPosition = [i, j]
                ray_origin, ray_direction = PixelToRay(camera_info, camera_pose, pixelPosition, args)
                ray_origins, delta = sampleRay(ray_origin, ray_direction, args)
                
                ray_directions = np.tile(ray_direction, (args.n_sample, 1))
    
                batch_rays_origin[i, j] = ray_origins
                batch_rays_direction[i, j] = ray_directions
                batch_delta[i, j] = delta

                color = image[pixelPosition[0], pixelPosition[1]]
                normalized_color = color / 255
                batch_pixelColors[i, j] = normalized_color
        
        
    ray_origins = torch.stack(batch_rays_origin).to(device)
    ray_directions = torch.stack(batch_rays_direction).to(device)
    batch_delta = torch.stack(batch_delta).to(device)
    pixel_colors = torch.stack(batch_pixelColors).to(device)
    
    return ray_origins, batch_delta, ray_directions, pixel_colors

def generateBatch2(images, poses, camera_info_list,mode,count, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """
    
    # randomly select a set of rays
    image_index = random.randint(0, images.shape[0]-1)
    camera_pose = poses[image_index]
    camera_info = camera_info_list[image_index]
    image = images[image_index]
    
    if mode == 'train':
        if count <args.precrop_iters:
            height, width = image.shape[:2]
            crop_size = (150, 150)  
            crop_x = max(0, (width - crop_size[0]) // 2)
            crop_y = max(0, (height - crop_size[1]) // 2)
            pixelPosition = np.random.randint(crop_y, crop_y + crop_size[1]-1, size=(args.n_rays_batch, 2))
        else:
            pixelPosition = np.random.randint(0, camera_info[0] - 1, size=(args.n_rays_batch, 2))
    else:  # mode == 'test'
        pixelPosition = np.array(np.meshgrid(range(camera_info[0]), range(camera_info[1]))).T.reshape(-1, 2)

    batch_rays_origin = np.empty((pixelPosition.shape[0], args.n_sample, 3))
    batch_rays_direction = np.empty((pixelPosition.shape[0], args.n_sample, 3))
    batch_delta = np.empty((pixelPosition.shape[0], args.n_sample))
    batch_pixelColors = np.empty((pixelPosition.shape[0], 3))

    for i, pixel_pos in enumerate(pixelPosition):
        ray_origin, ray_direction = PixelToRay(camera_info, camera_pose, pixel_pos, args)
        ray_origins, delta = sampleRay(ray_origin, ray_direction, args)

        ray_directions = np.tile(ray_direction, (args.n_sample, 1))

        batch_rays_origin[i] = ray_origins
        batch_rays_direction[i] = ray_directions
        batch_delta[i] = delta

        color = image[pixel_pos[0], pixel_pos[1]]
        normalized_color = color / 255
        batch_pixelColors[i] = normalized_color

    # Convert NumPy arrays to PyTorch tensors
    batch_rays_origin = torch.tensor(batch_rays_origin, dtype=torch.float32, device=device)
    batch_rays_direction = torch.tensor(batch_rays_direction, dtype=torch.float32, device=device)
    batch_delta = torch.tensor(batch_delta, dtype=torch.float32, device=device)
    batch_pixelColors = torch.tensor(batch_pixelColors, dtype=torch.float32, device=device)
    # Stack tensors
    ray_origins = torch.cat([batch_rays_origin], dim=1).to(device)
    ray_directions = torch.cat([batch_rays_direction], dim=1).to(device)
    pixel_colors = torch.cat([batch_pixelColors], dim=0).to(device)
    batch_delta = torch.cat([batch_delta], dim=0).to(device)
    
    
    return ray_origins, batch_delta, ray_directions, pixel_colors



def render(model, rays_origin, delta, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """
    rgb_data = []
    for i in range(rays_origin.shape[0]):
        rgb, sigma = model(rays_origin[i], rays_direction[i])
        
        alpha = 1 - torch.exp( -1 * sigma * delta[i])
        weights = torch.cumprod((1 - alpha  +  1e-10),-1) * alpha
        weights = weights.to(device)
        rgb_map = torch.sum(weights[...,None] * rgb, -2)
    
        rgb_data.append(rgb_map)

    rgb_data = torch.stack(rgb_data)  
        
    return rgb_data

def render2(model, rays_origin, delta, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """
    num_rays = rays_origin.shape[0]

    # Evaluate model for all rays in a single call
    rgb, sigma = model(rays_origin, rays_direction)

    # Calculate alpha values and weights
    alpha = 1 - torch.exp(-sigma * delta)
    weights = torch.cumprod((1 - alpha + 1e-10), dim=-1) * alpha
    weights = weights.to(device)

    # Calculate weighted sum of RGB values
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    # ipdb.set_trace()    

    # Stack the results along the first dimension
    # rgb_data = rgb_map.unsqueeze(0).expand(num_rays, -1, -1) 
        
    return rgb_map


def mse_loss(groundtruth, prediction):
    # criterion = nn.MSELoss()    
    # loss = criterion(groundtruth, prediction)
    loss = torch.mean((groundtruth - prediction) ** 2)
    return loss


def ssim_map(groundtruth, prediction):
    """
    Compute SSIM map between groundtruth and prediction using PyTorch.

    Parameters:
        groundtruth (torch.Tensor): Groundtruth image tensor.
        prediction (torch.Tensor): Predicted image tensor.
        window_size (int): Size of the SSIM window (default: 11).
        size_average (bool): If True, compute the average SSIM across all pixels (default: True).
        full (bool): If True, return SSIM map (default: False).

    Returns:
        float or torch.Tensor: SSIM value or SSIM map.
    """
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * 2) ** 2
    C2 = (K2 * 2) ** 2
    groundtruth = groundtruth.unsqueeze(0)
    prediction = prediction.unsqueeze(0)


    mu1 = torch.mean(groundtruth)
    mu2 = torch.mean(prediction)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.std(groundtruth)
    sigma2_sq = torch.std(prediction)
    sigma12 = torch.mean((groundtruth - mu1) * (prediction - mu2))


    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map



def findNearFar(image, pose, camera_info):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ray_origin, ray_direction = PixelToRay(camera_info, pose, [i, j], args)
            near = min(near, ray_origin[2])
            far = max(far, ray_origin[2])

def train(images, poses, camera_info, args):
    # initialize tensorboard, wandb, model, optimizer, scheduler
    Writer = SummaryWriter(args.logs_path)
    # wandb.init(project="Nerf-sr_f", 
    #            name="Nerf-sr_f1", # n_rays_batch, n_sample, precrop, save_ckpt_iter
    #            config=args)
    
    model = NeRFmodel(args.n_pos_freq, args.n_dirc_freq).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
    # decay_steps = args.lrate_decay * 1000
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lrate_decay)
    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iter = checkpoint['iter']
    else:
        start_iter = 0
        
    # find near and far range for each image in the dataset by projecting the 2D bounding box to 3D space using camera pose and camera matrix
    # for i in range(images.shape[0]):
    #     near, far = findNearFar(images[i], poses[i], camera_info[i])
        
    
    # start training
    for i in range(start_iter, args.max_iters):
        mode = 'train'  
        rays_origins, delta, rays_directions, ground_truth_rgb  = generateBatch2(images, poses, camera_info,mode,i, args)
        predicted_rgb = render2(model, rays_origins, delta, rays_directions, args)
        print("predicted_rgb",predicted_rgb)  
        mseloss = mse_loss(ground_truth_rgb, predicted_rgb)
        optimizer.zero_grad()
        mseloss.backward()
        optimizer.step()
        # if i % decay_steps == 0:   
        #     scheduler.step()
        if i % args.save_ckpt_iter == 0:
            checkpoint_path = args.checkpoint_path + "model_sr_f1_" + str(i) + ".pth"
            torch.save({
                'iter': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
        if i > 500 and i % 1000 == 0:
            test(images, poses, camera_info,model,i, args)
        psnr = 10 * torch.log10(1.0 / mseloss) 
        # wandb.log({'PSNR': float(psnr.item()), 'iter': i})
        Writer.add_scalar('PSNR', float(psnr.item()), i)   
        ssim_val = ssim_map(ground_truth_rgb, predicted_rgb)
        print("ssim_val",ssim_val)  
        Writer.add_scalar('SSIM', float(ssim_val.item()), i)
        # wandb.log({'SSIM': float(ssim_val.item()), 'iter': i})
        Writer.add_scalar('Loss', float(mseloss.item()), i)
        # wandb.log({'Loss': float(mseloss.item()), 'iter': i})
    Writer.close()  
    # wandb.finish()
    

def test(images, poses, camera_info,model,count, args):
    mode = 'test'
    print("test started")
    rays_origins, delta, rays_directions, ground_truth_rgb  = generateBatch2(images, poses, camera_info,mode,count, args)
    print("batch generated ")
    predicted_rgb = render2(model, rays_origins, delta, rays_directions, args)
    print("render done")
    predicted_rgb = predicted_rgb.reshape(images[0].shape[0],images[0].shape[1],3)
    plt.imsave(args.images_path + str(count) + '.png', predicted_rgb.detach().cpu().numpy())
    print("test1 done")


def main(args):
    # load data
    print("Loading data...")
    # os.chdir(os.path.dirname("cv_p02/Phase2/"))
    images, poses, camera_info = loadDataset(args.data_path, args.mode)

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Data/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--lrate_decay', type=int, default=250,help='exponential learning rate decay (in 1000s)')   # update 1
    # pre-crop options  : update 2
    parser.add_argument("--precrop_iters", type=int, default=8000,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    # near and far range    : update 3
    parser.add_argument('--near', type=int, default=2)
    parser.add_argument('--far',type=int, default=6)
    
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")  # 10
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction") #4
    parser.add_argument('--n_rays_batch',default=20*20,help="number of rays per batch") #32*32*4
    parser.add_argument('--n_sample',default=200,help="number of sample per ray")  #400
    parser.add_argument('--max_iters',default=100000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./checkpoints/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=False,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./images_sr_f1/",help="folder to store images")
    
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
    