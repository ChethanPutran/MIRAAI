import numpy as np
from scipy import linalg
import cv2 as cv

def homogeous_transformation_mat(R,t):
    """
    Returns the Homogenous transformation matrix from Rotaion matrix & traslation vector 

    Parameters:
    - R: Rotation matrix
    - t: Translation vector

    Returns:
    - H : Homogenous transformation matrix

    """

    H = np.eye(4)
    H[:3,:3] = R[:3,:3]
    H[:3, 3] = t[:3,0]
    return H

# Using epi-polar constraint
def triangulate_with_stereopsis(K,R,t, point_l, point_r):
    """
    Returns the 3D coordinate (X, Y, Z) in meters of a pixel using stereopsis method.
    
    Parameters:
    - P: Camera matrix for the left camera of the stero camera value at the pixel
    - point_l, point_r: pixel coordinates in the left & right image

    Returns:
    - (X, Y, Z): 3D coordinates in meters
    """
    baseline_m = 0.06  # baseline in meters (60mm)
    fx = K[0,0]
    fy = K[1,1]
    u0 = K[0, 2]
    v0 = K[1, 2]

    # disparity = x_l - x_r
    disparity = point_l[0]-point_r[0]
    print(disparity)

    X = np.ones((4,1))

    if disparity == 0:
        X[2,0] = float('inf')
        X[0,0] = point_l[0]
        X[1,0] = point_l[1]
        return X
    
    xL = point_l[0] - u0
    yL = point_l[1] - v0

    X[2,0] = (fx * baseline_m) / disparity  # Z
    X[0,0] = (xL * X[2,0]) / fx # X
    X[1,0] = (yL * X[2,0]) / fy # Y

    T = homogeous_transformation_mat(R,t)
    return T @ X

#direct linear transform
def triangulate_with_DLT(P1, P2, point1, point2):
    """
     Returns the 3D coordinate (X, Y, Z) in meters of a pixel using DLT method.
    
    Parameters:
    - P1 : Camera matrix for the camera of image pixel-point1
    - P2 : Camera matrix for the camera of image pixel-point2
    - point_l, point_r: pixel coordinates in the image1 & image2

    Returns:
    - (X, Y, Z): 3D coordinates in meters

    Implemetation:
    X_pixel x ( P * X_3d ) = 0 
    p1, p2, p3 - First,second and third rows of P respectively
    X - 3D cordinate of point 1 in left image & point 2 in right image
    v * (p3 . X) - (p2 . X) = 0
    (p1 . X) - u * (p3 . X) = 0
    u * (p2 . X) - v * (p1 . X) = 0

    Only first two equations are linearly independednt
    """

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    # U,S,V -> svd(A)
    # V,S,V' -> svd(A'A)
    # U,S,U' -> svd(AA')

    # B = A.transpose() @ A
    # U, s, Vh = linalg.svd(B, full_matrices = False)


    U,s,Vh = linalg.svd(A, full_matrices = False)
    # Find the last row of V for the smallest singular value, which is the best solutions for AX=0 up to scale.
    # Normalize the homogenous cordinate
    return Vh[3,0:3]/Vh[3,3]

def convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)


def get_camera_pose_in_world(rvec,tvec):
    """
     Returns the pose of camera w.r.to the world frame
    
    Parameters:
    - rvec : Rotation vector describes the transformation from the world frame to the camera frame.
    - tvec : Translation vector describes the transformation from the world frame to the camera frame.
    
    Note : rvec and tvec from calibrateCamera

    Returns:
    - R: Rotaion of camera w.r.to the world frame
    - t: Translation of camera w.r.to the world frame
    """ 

    # Convert rotation vector to rotation matrix
    R, _ = cv.Rodrigues(rvec)

    # Get inverse transformation: camera pose w.r.t world
    R_world = R.T  # inverse rotation
    t_world = -R_world @ tvec  # camera position in world

    return R_world,t_world