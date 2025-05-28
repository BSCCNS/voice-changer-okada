import numpy as np

def filter_points_in_box(data, box_min, box_max):
    """
    Filters rows in `data` that lie within the axis-aligned 3D bounding box.

    Parameters:
        data: (N, 3) NumPy array of 3D points
        box_min: (3,) array-like, minimum x, y, z values of the box
        box_max: (3,) array-like, maximum x, y, z values of the box

    Returns:
        filtered_data: (M, 3) array of points inside the box

        # Example usage
        box_min = [0, 0, 0]
        box_max = [10, 10, 10]
        filtered_data = filter_points_in_box(data, box_min, box_max)
    """
    data = np.asarray(data)
    box_min = np.asarray(box_min)
    box_max = np.asarray(box_max)

    in_box = np.all((data >= box_min) & (data <= box_max), axis=1)
    return data[in_box]

def project_points_to_box(data, box_min, box_max):
    """
    Projects 3D points to the nearest point within the box.
    
    Parameters:
        data: (N, 3) array of 3D points
        box_min: (3,) array-like, minimum x, y, z of the box
        box_max: (3,) array-like, maximum x, y, z of the box

    Returns:
        projected: (N, 3) array of points within the box
    """
    data = np.asarray(data)
    box_min = np.asarray(box_min)
    box_max = np.asarray(box_max)

    return np.clip(data, box_min, box_max)

# EMA SMOOTHENING
def exponential_moving_average(data, alpha=0.1):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed