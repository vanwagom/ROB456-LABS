#!/usr/bin/env python3

# This assignment lets you both define a strategy for picking the next point to explore and determine how you
#  want to chop up a full path into way points. You'll need path_planning.py as well (for calculating the paths)
#
# Note that there isn't a "right" answer for either of these. This is (mostly) a light-weight way to check
#  your code for obvious problems before trying it in ROS. It's set up to make it easy to download a map and
#  try some robot starting/ending points
#
# Given to you:
#   Image handling
#   plotting
#   Some structure for keeping/changing waypoints and converting to/from the map to the robot's coordinate space
#
# Slides

# The ever-present numpy
import numpy as np

# Your path planning code
import path_planning as path_planning
# Our priority queue
import heapq

# Using imageio to read in the image
import imageio

from scipy.ndimage import convolve


# -------------- Showing start and end and path ---------------
def plot_with_explore_points(im_threshhold, zoom=1.0, robot_loc=None, explore_points=None, best_pt=None):
    """Show the map plus, optionally, the robot location and points marked as ones to explore/use as end-points
    @param im - the image of the SLAM map
    @param im_threshhold - the image of the SLAM map
    @param robot_loc - the location of the robot in pixel coordinates
    @param best_pt - The best explore point (tuple, i,j)
    @param explore_points - the proposed places to explore, as a list"""

    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[0].set_title("original image")
    axs[1].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[1].set_title("threshold image")
    """
    # Used to double check that the is_xxx routines work correctly
    for i in range(0, im_threshhold.shape[1]-1, 10):
        for j in range(0, im_threshhold.shape[0]-1, 2):
            if is_reachable(im_thresh, (i, j)):
                axs[1].plot(i, j, '.b')
    """

    # Show original and thresholded image
    if explore_points is not None:
        for p in explore_points:
            axs[1].plot(p[1], p[0], '.b', markersize=1)

    for i in range(0, 2):
        if robot_loc is not None:
            axs[i].plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
        if best_pt is not None:
            axs[i].plot(best_pt[0], best_pt[1], '*y', markersize=10)
        axs[i].axis('equal')

    for i in range(0, 2):
        # Implements a zoom - set zoom to 1.0 if no zoom
        width = im_threshhold.shape[1]
        height = im_threshhold.shape[0]

        axs[i].set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
        axs[i].set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)


# -------------- For converting to the map and back ---------------
def convert_pix_to_x_y(im_size, pix, size_pix):
    """Convert a pixel location [0..W-1, 0..H-1] to a map location (see slides)
    Note: Checks if pix is valid (in map)
    @param im_size - width, height of image
    @param pix - tuple with i, j in [0..W-1, 0..H-1]
    @param size_pix - size of pixel in meters
    @return x,y """
    if not (0 <= pix[0] <= im_size[1]) or not (0 <= pix[1] <= im_size[0]):
        raise ValueError(f"Pixel {pix} not in image, image size {im_size}")

    return [size_pix * pix[i] / im_size[1 - i] for i in range(0, 2)]


def convert_x_y_to_pix(im_size, x_y, size_pix):
    """Convert a map location to a pixel location [0..W-1, 0..H-1] in the image/map
    Note: Checks if x_y is valid (in map)
    @param im_size - width, height of image
    @param x_y - tuple with x,y in meters
    @param size_pix - size of pixel in meters
    @return i, j (integers) """
    pix = [int(x_y[i] * im_size[1 - i] / size_pix) for i in range(0, 2)]

    if not (0 <= pix[0] <= im_size[1]) or not (0 <= pix[1] <= im_size[0]):
        raise ValueError(f"Loc {x_y} not in image, image size {im_size}")
    return pix


def is_reachable(im, pix):
    """ Is the pixel reachable, i.e., has a neighbor that is free?
    Used for
    @param im - the image
    @param pix - the pixel i,j"""

    # Returns True (the pixel is adjacent to a pixel that is free)
    #  False otherwise
    # You can use four or eight connected - eight will return more points
    # YOUR CODE HERE

    # Init start bool to False
    reachable = False

    # If any of the 8 neighbors are free, return True
    for adj_node in path_planning.eight_connected(pix):
        if path_planning.is_free(im, adj_node) is True:
            reachable = True

    return reachable


def find_all_possible_goals(im):
    """Find all pixels that are unknown and adjacent to a free pixel.
    @param im: Thresholded image (numpy array).
    @return: List of possible goal pixel coordinates (tuples).
    """

    # List of 8 neighbor offsets
    neighbors = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (x, y) != (0, 0)]

    # Create mask for unknown pixels
    unknown_value = 128
    unknown_mask = (im == unknown_value)

    # Create mask for free pixels
    free_mask = (im == 255)

    # Init a mask for valid neighbors
    neighbor_mask = np.zeros_like(im, dtype=bool)

    # Roll free pixel neighbors
    for di, dj in neighbors:
        neighbor_mask = np.roll(np.roll(free_mask, di, axis=0), dj, axis=1)

    # Combine masks to find unknown pixels next to free pixels
    possible_goal_mask = neighbor_mask & unknown_mask

    # Find coordinates of possible goal pixels
    possible_goals = np.argwhere(possible_goal_mask)

    # Filter using is_reachable
    possible_goals = [tuple(coord) for coord in possible_goals if is_reachable(im, tuple(coord))]

    return possible_goals


def find_best_point(im, possible_points, robot_loc, min_area=0.5):
    """
    Pick the best unseen point to go to, considering proximity and connectivity.
    
    @param im: Thresholded image (numpy array).
    @param possible_points: List of possible points to choose from.
    @param robot_loc: Location of the robot as a tuple (x, y).
    @param min_area: Minimum precentage around a point that must also be possible.
    @return: The best point as a tuple (x, y).
    """

    # Create a mask for the unknown pixels
    unknown_mask = (im == 128)

    # Create the kernal that will be used to find the number of adjacent pixels that are unknown
    kernal = np.ones((5, 5), dtype=int)

    # Calculate the minimum number of adjacent pixels that must be unknown for the pixel to pass the filter
    min_pixels = int((np.size(kernal) - 1) * min_area)

    # Convolve the unknown mask with the kernal to find the number of adjacent pixels that are unknown. Creates an
    # array where the values represent the number of neighbors that are unknown
    connectivity = convolve(unknown_mask.astype(int), kernal, mode='constant', cval=0)

    # Create a list of valid points that are unknown and have enough neighbors that are also unknown
    valid_points = []
    for point in possible_points:
        x, y = point
        if im[y, x] == 128 and connectivity[y, x] >= min_pixels:
            valid_points.append(point)

    if not valid_points:
        return None

    # Init best point stored as a tuple (distance, point)
    best_point = [np.inf, (0, 0)]

    # Iterate over the possible points and search for the closest unknown point to the robot
    for point in valid_points:

        # Calculate the distance from the robot to the point
        dist = np.sqrt((point[0] - robot_loc[0]) ** 2 + (point[1] - robot_loc[1]) ** 2)

        # If the distance is less than the current best point, update the best point
        if dist < best_point[0]:
            best_point = [dist, point]

    if best_point[1] == (0, 0):
        return None

    return best_point[1]


def find_waypoints(im, path, res=0.95):
    """ Place waypoints along the path
    @param im - the thresholded image
    @param path - the initial path
    @ return - a new path"""

    # Split the path equally into points
    split_path = []
    for i, point in enumerate(path):
        if i % 4 == 0:
            split_path.append(point)

    corners = []
    for i in range(0, len(split_path) - 1):

        # Get the vectors around the central point
        prev_point = split_path[i - 1]
        current_point = split_path[i]
        next_point = split_path[i + 1]

        # Calculate the angle between vectors at the point
        prev_vector = (current_point[0] - prev_point[0], current_point[1] - prev_point[1])
        next_vector = (next_point[0] - current_point[0], next_point[1] - current_point[1])
        curvature = np.dot(prev_vector, next_vector) / (np.linalg.norm(prev_vector) * np.linalg.norm(next_vector))

        # If the curvature is not 1, then we have a corner
        if curvature < res:
            corners.append(split_path[i])

    # Add the goal location back into the list
    corners.append(path[-1])

    return corners


if __name__ == '__main__':
    # Doing this here because it is a different yaml than JN
    import yaml_1 as yaml

    im, im_thresh = path_planning.open_image("map.pgm")

    robot_start_loc = (1940, 1953)

    all_unseen = find_all_possible_goals(im_thresh)
    best_unseen = find_best_point(im_thresh, all_unseen, robot_loc=robot_start_loc)

    plot_with_explore_points(im_thresh, zoom=0.1, robot_loc=robot_start_loc, explore_points=all_unseen,
                             best_pt=best_unseen)

    path = path_planning.dijkstra(im_thresh, robot_start_loc, best_unseen)
    waypoints = find_waypoints(im_thresh, path)
    path_planning.plot_with_path(im, im_thresh, zoom=0.1, robot_loc=robot_start_loc, goal_loc=best_unseen,
                                 path=waypoints)

    # Depending on if your mac, windows, linux, and if interactive is true, you may need to call this to get the plt
    # windows to show
    # plt.show()

    print("Done")
