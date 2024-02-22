import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def error_function(x, X_points, feature_points,K ):

    R = Rotation.from_quat(x[0:4]).as_matrix()
    R = R.reshape((3, 3))
    t = x[4:7]
    t = t.reshape((3, 1))
    Transformation_mat = np.append(R, -t, axis=1).tolist() 
    Transformation_mat.append([0,0,0,1])         # [3 x 4] matrix
    Transformation_mat = np.array(Transformation_mat)
    d1= np.array([0,0,0])
    K=np.append(K,d1.reshape((3, 1)) , axis=1)
    Projection_left = K @ Transformation_mat

    errors = []

    

    u = feature_points[:, 0]
    v = feature_points[:, 1]

    ones = np.ones((X_points.shape[0], 1))
    world_points_homogen = np.concatenate((X_points, ones), axis=1)
    proj = np.matmul(Projection_left, world_points_homogen.T)
    proj = proj / proj[-1]
    proj = proj.T
    for i in range(len(proj)):
        error = np.sqrt((u[i] - proj[i][0])**2 + (v[i] - proj[i][1])**2)
        errors.append(error)

    error_total = np.mean(np.array(error).squeeze())
    print("error_total", error_total)

    return error_total

def get_homogenous_coordinates(coordinates):
    """
    Input: The co-ordinates u,v
    Outputs : The homogenize coordinates
    """

    coordinates = np.asarray(coordinates)
    ones = np.ones((coordinates.shape[0], 1))

    homo = np.concatenate((coordinates, ones), axis=1)

    return homo

def nonlinear_PnP(K, feature_points, X_points, R, C):

    QuaterParams = Rotation.from_matrix(R).as_quat()

    X = [QuaterParams[0], QuaterParams[1], QuaterParams[2], QuaterParams[3], C[0], C[1], C[2]]

    optimized_params = least_squares(fun=error_function, x0=X, method="trf", args=[X_points, feature_points, K])

    X_opt = optimized_params.x

    QuaterParams = X_opt[:4]
    t = X_opt[4:]
    R = Rotation.from_quat(QuaterParams).as_matrix()

    t = t.reshape((3, 1))                           # make into column vector
    Transformation_mat = np.append(R, -t, axis=1).tolist() 
    Transformation_mat.append([0,0,0,1])         # [3 x 4] matrix
    Transformation_mat = np.array(Transformation_mat)

    d1= np.array([0,0,0])
    K=np.append(K,d1.reshape((3, 1)) , axis=1)
    Projection_left = K @ Transformation_mat
    Projection_right= K

    return  C,R , Projection_right, Projection_left
