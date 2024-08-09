#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import cv2
from plyfile import PlyData, PlyElement


# In[25]:


def image_derivatives(image):
    kernel = np.array([-1, 0, 1])
    Ix = cv2.filter2D(image, -1, kernel)
    Iy = cv2.filter2D(image, -1, kernel.T)
    return Ix, Iy


def gaussian_smoothing(derivatives):
    kernel = cv2.getGaussianKernel(5, -1)
    Ix_smoothed = cv2.sepFilter2D(derivatives[0], -1, kernel, kernel.T)
    Iy_smoothed = cv2.sepFilter2D(derivatives[1], -1, kernel, kernel.T)
    return Ix_smoothed, Iy_smoothed


def harris_response(Ix_smoothed, Iy_smoothed):
    Ixx = Ix_smoothed**2
    Iyy = Iy_smoothed**2
    Ixy = Ix_smoothed * Iy_smoothed

    det_M = Ixx * Iyy - Ixy**2
    trace_M = Ixx + Iyy
    k = 0.04

    return det_M - k * trace_M**2

def non_maximum_suppression(harris_response):
    response_filtered = cv2.dilate(harris_response, np.ones((3, 3)))
    maxima = (harris_response == response_filtered)
    return maxima


def strongest_corners(harris_response, maxima, num_corners=100):
    harris_max = harris_response * maxima
    idx = np.argsort(harris_max.ravel())[::-1][:num_corners]
    return np.column_stack(np.unravel_index(idx, harris_response.shape))


#     def corners_to_3D_points(corners, depth_map, K, S=5000):
#         num_corners = corners.shape[0]
#         points_3D = np.zeros((num_corners, 3))

#         for i, (x, y) in enumerate(corners):
#             d = depth_map[y, x]
#             if d == 0:
#                 continue
#             point_2D = np.array([x, y, 1])
#             point_3D = d / S * np.linalg.inv(K) @ point_2D
#             points_3D[i] = point_3D

#         valid_points = points_3D[~np.all(points_3D == 0, axis=1)]
#         return valid_points

#         def corners_to_3D_points(corners, depth_map, K, S=5000):
#             inverse_k = np.linalg.inv(K)
#             _3d_points = {}
#             for corner in corners:
#                 x, y = corner
#                 d = depth[x, y]
#                 if d == 0:
#                     continue
#                 harris_2d = [y, x, 1]
#                 r = (1/s) * d * np.dot(inverse_k, harris_2d)
#                 valid_points[corner] = r

#             return valid_points

def corners_to_3D_points(corners, depth_map, K, S=1):
    points = []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for corner in corners:
        x, y = corner
        d = depth_map[y, x]  # Corrected variable access from 'depth_map[x, y]' to 'depth_map[y, x]'
        if d == 0:
            continue
        X = (x - cx) * d / fx
        Y = (y - cy) * d / fy
        Z = d
        points.append([X, Y, Z])
    return np.array(points) * S


def rank_transform(image, window_size=5):
    half_window = window_size // 2
    img_padded = cv2.copyMakeBorder(image, half_window, half_window, half_window, half_window, cv2.BORDER_REFLECT)
    ranks = np.zeros_like(image, dtype=np.uint8)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            window = img_padded[y:y + window_size, x:x + window_size]
            ranks[y, x] = np.sum(window < window[half_window, half_window])

    return ranks

def compute_distances(corners1, corners2, rank1, rank2, window_size=11):
    half_window = window_size // 2
    num_corners1, num_corners2 = corners1.shape[0], corners2.shape[0]
    distances = np.zeros((num_corners2, num_corners1), dtype=np.float32)
    rank1_padded = cv2.copyMakeBorder(rank1, half_window, half_window, half_window, half_window, cv2.BORDER_REFLECT)
    rank2_padded = cv2.copyMakeBorder(rank2, half_window, half_window, half_window, half_window, cv2.BORDER_REFLECT)

    for i, (y2, x2) in enumerate(corners2):
        for j, (y1, x1) in enumerate(corners1):
            window1 = rank1_padded[y1:y1 + window_size, x1:x1 + window_size]
            window2 = rank2_padded[y2:y2 + window_size, x2:x2 + window_size]
            distances[i, j] = np.sum(np.abs(window1 - window2))

    return distances

def find_best_matches(distances, threshold=0.7):
    matches = []
    for i in range(distances.shape[1]):
        min_idx = np.argmin(distances[:, i])
        min_val = distances[min_idx, i]
        distances[min_idx, i] = np.inf
        second_min_val = distances[:, i].min()

        if min_val < threshold * second_min_val:
            matches.append((min_idx, i))

    return matches


# In[26]:


# Pose Estimation:

def estimate_pose(matches, points1_3D, points2_3D):
    num_matches = len(matches)
    P1 = np.zeros((num_matches, 3))
    P2 = np.zeros((num_matches, 3))

    for i, (pt2, pt1) in enumerate(matches):
        P1[i] = points1_3D[pt2]
        P2[i] = points2_3D[pt1]

    mean1, mean2 = np.mean(P1, axis=0), np.mean(P2, axis=0)
    P1_centered, P2_centered = P1 - mean1, P2 - mean2
  
    W = P2_centered.T @ P1_centered
    U, _, Vt = np.linalg.svd(W)
    R = U @ Vt
    t = mean1 - R @ mean2

    return R, t


# In[28]:


# Part 5: Finis Coronat Opus
def transform_points(points, R, t):

    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    #transformed_points_hom = R @ points_hom.T + t[:3, None]
    transformed_points_hom = R @ points_hom[:, :3].T + t[:3, None]
    #transformed_points = (transformed_points_hom / transformed_points_hom[-1, :])[:3, :].T
    epsilon = 1e-8
    transformed_points = (transformed_points_hom / (transformed_points_hom[-1, :] + epsilon))[:3, :].T

    return transformed_points

def get_colored_3d_points(image, depth_map, K, S=5000):
    height, width, _ = image.shape
    points_3d = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            d = depth_map[y, x]
            if d == 0:
                continue
            point_2d = np.array([x, y, 1])
            point_3d = d / S * np.linalg.inv(K) @ point_2d
            points_3d[y, x] = point_3d
    valid_points = points_3d[~np.all(points_3d == 0, axis=2)]
    valid_colors = image[~np.all(points_3d == 0, axis=2)]
    return valid_points, valid_colors

# Load the images
image1 = cv2.imread("rgb1.png")
image2 = cv2.imread("rgb2.png")
image3 = cv2.imread("rgb3.png")

depth1 = cv2.imread("depth1.png", cv2.IMREAD_ANYDEPTH)
depth2 = cv2.imread("depth2.png", cv2.IMREAD_ANYDEPTH)
depth3 = cv2.imread("depth3.png", cv2.IMREAD_ANYDEPTH)

if image1 is None or image2 is None or image3 is None:
    print("Error loading images.")
else:
    # Find corners and rank transforms for images 1, 2, and 3
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

    hr1 = harris_response(*gaussian_smoothing(image_derivatives(gray1)))
    hr2 = harris_response(*gaussian_smoothing(image_derivatives(gray2)))
    hr3 = harris_response(*gaussian_smoothing(image_derivatives(gray3)))

    corners1 = strongest_corners(hr1, non_maximum_suppression(hr1))
    corners2 = strongest_corners(hr2, non_maximum_suppression(hr2))
    corners3 = strongest_corners(hr3, non_maximum_suppression(hr3))

    rank1 = rank_transform(gray1)
    rank2 = rank_transform(gray2)
    rank3 = rank_transform(gray3)

    # Compute the distances between corners of images 1 and 2, and find best matches
    distances12 = compute_distances(corners1, corners2, rank1, rank2)
    matches12 = find_best_matches(distances12)

    # Similarly, compute the distances between corners of images 2 and 3, and find best matches
    distances23 = compute_distances(corners2, corners3, rank2, rank3)
    matches23 = find_best_matches(distances23)

    points1_3D = corners_to_3D_points(corners1[:, ::-1], depth1, K)
    points2_3D = corners_to_3D_points(corners2[:, ::-1], depth2, K)
    points3_3D = corners_to_3D_points(corners3[:, ::-1], depth3, K)


    # Estimate the pose between images 1 and 2, and between images 2 and 3
    R12, t12 = estimate_pose(matches12, points1_3D, points2_3D)
    R23, t23 = estimate_pose(matches23, points2_3D, points3_3D)

    points1_3D_full, colors1_full = get_colored_3d_points(image1, depth1, K)
    points2_3D_full, colors2_full = get_colored_3d_points(image2, depth2, K)
    points3_3D_full, colors3_full = get_colored_3d_points(image3, depth3, K)
    
    # added later
    points1_3D_full = points1_3D_full[np.isfinite(points1_3D_full).all(axis=1)]
    points2_3D_full = points2_3D_full[np.isfinite(points2_3D_full).all(axis=1)]
    points3_3D_full = points3_3D_full[np.isfinite(points3_3D_full).all(axis=1)]

    points1_transformed_full = transform_points(points1_3D_full, R12, t12)
    points3_transformed_full = transform_points(points3_3D_full, R23.T, -R23.T @ t23)


    # Transform the points from the third image to the coordinate system of the second image (new Code)
    points3_transformed_to_2 = transform_points(points3_3D_full, R23.T, -R23.T @ t23)

    # Transform the points from the second image (including the transformed points from the third image) to the coordinate system of the first image
    points2_transformed_to_1 = transform_points(points2_3D_full, R12, t12)#(new code)
    points3_transformed_to_1 = transform_points(points3_transformed_to_2, R12, t12)#(new code)    
    
    def save_points_to_ply(points, colors, filename):
        num_points = len(points)
        points_colors = np.hstack([points, colors]).reshape(-1, 6)

        vertex = np.array([tuple(pc) for pc in points_colors],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        ply_header = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])], text=True, byte_order='<')
        ply_header.write(filename)
    
    save_points_to_ply(points1_3D_full, colors1_full, 'output11.ply')
    save_points_to_ply(points2_3D_full, colors2_full, 'output22.ply')
    save_points_to_ply(points3_3D_full, colors3_full, 'output33.ply')
    
    # Combine the transformed points from the second and third images with the points from the first image
    combined_points = np.vstack([points1_3D_full, points2_transformed_to_1, points3_transformed_to_1])
    
    #final_output = np.vstack([points1_3D_full, points2_3D_full, points3_3D_full])
    combined_colors = np.vstack([colors1_full, colors2_full, colors3_full])
    
    # Save the combined point cloud to a PLY file
    save_points_to_ply(combined_points, combined_colors, 'combined_output11.ply')

    

