import cv2
import numpy as np
import open3d as o3d
from scipy.signal import convolve2d


def project_lidar_to_2d(lidar_cloud, labels, delta_theta=0.08, delta_phi=0.4, mode="distance"):
    """Projects a 3D LiDAR point cloud into a 2D map for depth or semantic information based on angular resolution.

    Parameters
    ----------
    - lidar_cloud: An Open3D point cloud object.
    - delta_theta: Horizontal angular resolution in degrees.
    - delta_phi: Vertical angular resolution in degrees.
    - mode: 'depth' for depth map, 'semantic' for semantic map.

    Returns
    -------
    - A 2D numpy array representing the 2D map.

    """
    # Convert point cloud to numpy array
    points = np.asarray(lidar_cloud.points)

    # Calculate the azimuth and elevation angles
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # Azimuth
    phi = np.arcsin(z / r)  # Elevation

    # Convert angular resolution from degrees to radians
    delta_theta = np.deg2rad(delta_theta)
    delta_phi = np.deg2rad(delta_phi)

    # Calculate indices for the 2D map
    theta_indices = np.floor((theta + np.pi) / delta_theta).astype(int)
    phi_indices = np.floor((phi + (np.pi / 2)) / delta_phi).astype(int)

    # Create the 2D map
    max_theta_index = int(np.ceil(2 * np.pi / delta_theta))
    max_phi_index = int(np.ceil(np.pi / delta_phi))
    map2d = np.zeros((max_phi_index, max_theta_index), dtype=np.uint8)
    index_map = np.full((max_phi_index, max_theta_index), -1, dtype=int)  # -1 indicates no point mapped

    # Fill in the map values
    mapped_points = 0
    for index, d in enumerate(r):
        col = theta_indices[index]
        row = phi_indices[index]
        # Decide what value to place based on the mode
        if mode == "distance":
            current_value = map2d[row, col]
            if current_value == 0 or d < current_value:
                map2d[row, col] = d
                index_map[row, col] = index  # Store the original point index
                mapped_points += 1
        elif mode == "semantic":
            # For semantic mode, use RGB color
            current_value = map2d[row, col]
            if current_value == 0 or d < current_value:
                map2d[row, col] = labels[index] * 10  # Assign color if closer
                index_map[row, col] = index  # Store the original point index
                mapped_points += 1

    return map2d, index_map


def fill_checkerboard(image):
    """Fill a checkerboard patterned image by averaging non-zero neighbors.

    Parameters
    ----------
    - image: A 2D numpy array of the image with zeros in a checkerboard pattern.

    Returns
    -------
    - A filled numpy array.

    """
    # Create a kernel that considers the eight surrounding cells
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # Apply convolution to count non-zero neighbors
    neighbors = convolve2d((image > 0).astype(int), kernel, mode="same", boundary="fill", fillvalue=0)

    # Sum of non-zero neighbor values
    neighbor_sum = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        filled_image = np.where(image > 0, image, neighbor_sum / neighbors)
        filled_image[neighbors == 0] = 0  # For cells with no non-zero neighbors

    return filled_image.astype(np.uint8)


def detect_and_visualize_contours(image, mode="distance", delta=10):
    """Detects and visualizes contours in a LiDAR-derived image using OpenCV,
    focusing on changes in pixel intensity greater than a specified delta.

    Parameters
    ----------
    - image: 2D numpy array of the image.
    - mode: 'distance' for grayscale distance images, 'semantic' for semantic label images.
    - delta: The minimum change in intensity for an edge to be recognized.

    """
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert image to a suitable format for OpenCV
    norm_image = fill_checkerboard(
        fill_checkerboard(norm_image)
    )  # cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Sobel operators to get the gradient magnitude
    grad_x = cv2.Sobel(norm_image, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(norm_image, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Threshold the gradient image
    if mode == "distance":
        _, thresh_image = cv2.threshold(gradient, delta, 255, cv2.THRESH_BINARY)
    elif mode == "semantic":
        thresh_image = cv2.adaptiveThreshold(
            gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        raise ValueError("Invalid mode. Use 'distance' or'semantic'.")

    # Find contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def map_contours_to_point_indices(contours, index_map):
    # Extract contours from the image
    point_indices = []

    # Map contours back to original point indices
    for contour in contours:
        for point in contour:
            y, x = point[0][1], point[0][0]
            point_index = index_map[y, x]
            if point_index != -1:  # Only add valid indices
                point_indices.append(point_index)

    return point_indices


def highlight_contour_points(cloud, contour_indices, highlight_color=[1, 0, 0]):  # Red color
    """Highlights the contour points in the point cloud.

    Parameters
    ----------
    - cloud: Open3D point cloud object.
    - contour_indices: Indices of the points that are part of contours.
    - highlight_color: Color to apply to contour points, default is red.

    Returns
    -------
    - A new Open3D point cloud object with highlighted contour points.

    """
    # Create a copy of the cloud to modify
    highlighted_cloud = cloud  # .clone()

    # Convert colors to a mutable numpy array
    colors = np.asarray(highlighted_cloud.colors)

    # Highlight the contour indices
    for idx in contour_indices:
        colors[idx] = highlight_color

    # Update the colors in the point cloud
    highlighted_cloud.colors = o3d.utility.Vector3dVector(colors)

    return highlighted_cloud


def find_contours(cloud, labels, mode="distance", delta=10):
    lidar_img, index_map = project_lidar_to_2d(cloud, labels, mode=mode)
    lidar_img = lidar_img.astype(np.uint8)

    contours = detect_and_visualize_contours(lidar_img, mode=mode, delta=delta)

    point_indices = map_contours_to_point_indices(contours, index_map)

    # highlighted_cloud = highlight_contour_points(cloud, point_indices)

    contour_points = np.asarray(cloud.points)[point_indices]
    contour_colors = np.asarray(cloud.colors)[point_indices]
    contour_cloud = o3d.geometry.PointCloud()
    contour_cloud.points = o3d.utility.Vector3dVector(contour_points)
    contour_cloud.colors = o3d.utility.Vector3dVector(contour_colors)
    # o3d.visualization.draw_geometries([highlighted_cloud])
    return contour_cloud
