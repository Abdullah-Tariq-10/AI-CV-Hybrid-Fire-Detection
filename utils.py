import numpy as np
import cv2

def calculate_radial_distances(contour, centroid, num_rays=12):
    """
    Calculates radial distances from the centroid to the contour at specified ray intervals.
    We compute distances by casting rays from the centroid and finding the maximum distance
    to a contour point that falls within the angular bin.
    """
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    distances = []
    
    cx, cy = centroid
    contour_points = contour.reshape(-1, 2)
    diffs = contour_points - [cx, cy]
    
    # Calculate angle and distance of each contour point relative to centroid
    pts_angles = np.arctan2(diffs[:, 1], diffs[:, 0]) # [-pi, pi]
    pts_angles = (pts_angles + 2 * np.pi) % (2 * np.pi) # [0, 2*pi]
    pts_dists = np.linalg.norm(diffs, axis=1)
    
    # 15 degree tolerance on either side of the ray to capture points in the bin
    tol = np.radians(15)
    
    for ray_angle in angles:
        # Calculate angular difference handling the 2*pi wrap-around
        angle_diff = np.abs(pts_angles - ray_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        
        valid_idx = angle_diff < tol
        
        if np.any(valid_idx):
            max_dist = np.max(pts_dists[valid_idx])
            distances.append(max_dist)
        else:
            distances.append(0.0)
            
    return distances

def draw_radial_distances(frame, centroid, distances, num_rays=12):
    """
    Draws the computed radial distances on the frame for visualization.
    """
    cx, cy = centroid
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    for i, angle in enumerate(angles):
        dist = distances[i]
        if dist > 0:
            px = int(cx + dist * np.cos(angle))
            py = int(cy + dist * np.sin(angle))
            cv2.line(frame, (cx, cy), (px, py), (255, 0, 0), 1)
