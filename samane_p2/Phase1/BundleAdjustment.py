import os
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import cv2  
import ipdb 

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def bundle_adjustment(n_cameras, n_points, camera_params, points_3d, camera_indices, point_indices, points_2d):
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.plot(f0)
    plt.show()
    import time
    start = time.time()
    import scipy.optimize
    max_iterations = 100
    res = scipy.optimize.least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(n_cameras, n_points, camera_indices, point_indices, points_2d), max_nfev=max_iterations)
    print("time", time.time()-start)
    plt.plot(res.fun)
    plt.show()
    camera_poses_optimized = []
    cc= res.x[:n_cameras * 9].reshape((n_cameras, 9))
    for i in range(n_cameras):
        camera_params = cc[i]
        intrinsic_mat = np.array([[camera_params[6], 0, camera_params[7]], [0, camera_params[6], camera_params[8]], [0, 0, 1]])
        R, _ = cv2.Rodrigues(camera_params[:3])
        t = camera_params[3:6]
        camera_poses_optimized.append([t, R])
    world_points_optimized = res.x[n_cameras*9:].reshape((n_points, 3))
    

    return camera_poses_optimized, world_points_optimized