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

def generateBatch(images, poses, camera_info, args):
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
    batch_rays_origin = []
    batch_rays_direction = []
    batch_delta = []
    batch_pixelColors = []
    for i in range(camera_info[0]):
        for j in range(camera_info[1]):
            # randomly select a pixel
            pixelPosition = [i, j]
            ray_origin, ray_direction = PixelToRay(camera_info, camera_pose, pixelPosition, args)
            ray_origins, delta = sampleRay(ray_origin, ray_direction, args)
            
            ray_directions = ray_direction.expand(args.n_sample, 3)
            
            batch_rays_origin.append(ray_origins)
            batch_rays_direction.append(ray_directions)
            batch_delta.append(delta)

            color = image[pixelPosition[0], pixelPosition[1]]
            normalized_color = color / 255
            batch_pixelColors.append(normalized_color)
        
    ray_origins = torch.stack(batch_rays_origin).to(device)
    ray_directions = torch.stack(batch_rays_direction).to(device)
    batch_delta = torch.stack(batch_delta).to(device)
    pixel_colors = torch.stack(batch_pixelColors).to(device)
    
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
    # for i in range(rays_origin.shape[0]):
    #     rgb, sigma = model(rays_origin[i], rays_direction[i])
    #     # alpha = 1 - torch.exp(-sigma * delta[i])
    #     # weights = torch.exp(-torch.cumsum(sigma[:-1] * delta[i][:1], dim=0))
    #     # weights = torch.cat((torch.tensor([0]).to(device), weights)).expand((1,400))
    
    #     # rgb = torch.sum(weights.T * rgb, dim=0)
    #     # weights_sum = torch.sum(weights, dim=0)
    #     sigma = sigma.unsqueeze(1)
    #     delta_current = delta[i].unsqueeze(1)
    #     rgb_map = render_rays2(rgb, sigma, delta_current,  rays_origin, rays_direction)
    #     rgb_data.append(rgb_map)
    #     print("rgb_map",rgb_map)
    # ipdb.set_trace()  
    for i in range(rays_origin.shape[0]):
        rgb, sigma = model(rays_origin[i], rays_direction[i])
        
        alpha = 1 - torch.exp(-sigma * delta[i])
        weights = torch.exp(-torch.cumsum(sigma * delta[i], dim=0))
        # weights = torch.cat((torch.tensor([0]), weights))
        rgb_map = torch.sum(weights.unsqueeze(1) * rgb, dim=0) / (torch.sum(weights) + 1e-10)
        rgb_data.append(rgb_map)
        # print("rgb_map", rgb_map)

    rgb_data = torch.stack(rgb_data)  
        
    return rgb_data

def render_rays2(rgb_list, depth_list, delta_list,  ray_origin, ray_direction):
    """ As per the paper""" 
    #print("start")  
    #print(rgb_list[100]  )  
    alpha = 1 - torch.exp(-1* depth_list * delta_list )
    #print(alpha)
    #print(alpha[10])
    weights = compute_transmittance(1 - alpha)  * alpha
    #print("*****")
    #print(weights)
    pixel_value = torch.sum(weights * rgb_list, 0)
    # print(pixel_value)    
    return pixel_value   


def compute_transmittance(alpha):    
    transmittance = torch.cumprod(alpha,0)
    #print(transmittance)
    #accu_transmittance = torch.cat((torch.ones((transmittance.shape[0]), device = alpha.device), 
    #            transmittance[:]), dim=-1)
    
    accu_transmittance = torch.roll(transmittance , shifts = 1,  dims = 1)
    #print(accu_transmittance.shape)
    accu_transmittance[0,0] = 1
    #print(accu_transmittance)
    
    return accu_transmittance


def mse_loss(groundtruth, prediction):
    # criterion = nn.MSELoss()    
    # loss = criterion(groundtruth, prediction)
    loss = torch.mean((groundtruth - prediction) ** 2)
    return loss

def ssim_map(groundtruth, prediction):
    pass

def findNearFar(image, pose, camera_info):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ray_origin, ray_direction = PixelToRay(camera_info, pose, [i, j], args)
            near = min(near, ray_origin[2])
            far = max(far, ray_origin[2])

def train(images, poses, camera_info, args):
    # initialize tensorboard, wandb, model, optimizer, scheduler
    Writer = SummaryWriter(args.logs_path)
    wandb.init(project="NeRF", 
               name="NeRF",
               config=args)
    
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
        rays_origins, delta, rays_directions, ground_truth_rgb  = generateBatch(images, poses, camera_info, args)
        
        predicted_rgb = render(model, rays_origins, delta, rays_directions, args)
        print("predicted_rgb",predicted_rgb)  
        mseloss = mse_loss(ground_truth_rgb, predicted_rgb)
        optimizer.zero_grad()
        mseloss.backward()
        optimizer.step()
        # if i % decay_steps == 0:   
        #     scheduler.step()
        # if i % args.save_ckpt_iter == 0:
        #     torch.save({
        #         'iter': i,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict()
        #     }, args.checkpoint_path)
        if i % 5 == 0:
            predicted_rgb = predicted_rgb.reshape(50,50,3)
            plt.imsave(args.images_path + str(i) + '.png', predicted_rgb.detach().cpu().numpy())
        psnr = 10 * torch.log10(1.0 / mseloss) 
        print("psnr",psnr)
        wandb.log({'PSNR': float(psnr.item()), 'iter': i})
        Writer.add_scalar('PSNR', float(psnr.item()), i)   
        # ssim_val = ssim_map(images, predicted_rgb)
        # Writer.add_scalar('SSIM', float(ssim_val.item()), i)
        # wandb.log({'SSIM': float(ssim_val.item()), 'iter': i})
        Writer.add_scalar('Loss', float(mseloss.item()), i)
        wandb.log({'Loss': float(mseloss.item()), 'iter': i})
    Writer.close()  
    wandb.finish()
    

def test(images, poses, camera_info, args):
    pass


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
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    # near and far range    : update 3
    parser.add_argument('--near', type=int, default=2)
    parser.add_argument('--far',type=int, default=6)
    
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=50*50,help="number of rays per batch") #32*32*4
    parser.add_argument('--n_sample',default=200,help="number of sample per ray")  #400
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./checkpoints",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=False,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./images_m2/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
    