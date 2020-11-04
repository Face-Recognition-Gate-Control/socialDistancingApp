import pyrealsense2 as rs
import math
import core.detection.config_caffe as config
import cv2
import numpy as np


def drawBox(image, predicitons, min_dist):
    violation = set()

    overlay = image.copy()
    output = image.copy()
    alpha = 0.25

    if len(predicitons[1]) >= 2:

        violation = euclideanDistance(predicitons[1], min_dist, predicitons[2], image)

    for (i, (box)) in enumerate(predicitons[0]):

        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        (startX, startY, endX, endY) = box

        color = (50, 205, 50)
        if i in violation:
            color = (0, 0, 255)
        cv2.rectangle(overlay, (startX, startY), (endX, endY), color, -1)
        w = startX + (endX - startX) / 2
        h = startY + (endY - startY) / 2

        cv2.circle(overlay, (int(w), int(h)), 10, color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output, violation


def compute_perspective_transform(corner_points, width, height, image):
    """Compute the transformation matrix
    @ corner_points : 4 corner points selected from the image
    @ height, width : size of the image
    return : transformation matrix and the transformed image
    """
    # Create an array out of the 4 corner points
    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed


def compute_point_perspective_transformation(matrix, list_downoids):
    """Apply the perspective transformation to every ground point which have been detected on the main frame.
    @ matrix : the 3x3 matrix
    @ list_downoids : list that contains the points to transform
    return : list containing all the new points
    """
    # Compute the new coordinates of our points
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append(
            [transformed_points[i][0][0], transformed_points[i][0][1]]
        )
    return transformed_points_list


def distancePixel(points):

    if len(points) >= 2:

        violate = set()

        for i in range(0, len(points)):

            for j in range(i + 1, len(points)):

                dist = calculate_distance_of_two_points_of_boxes(points[i], points[j])
                dist = dist / 100
                print(dist)
                if dist < 1:

                    violate.add(i)
                    violate.add(j)

        return violate


def calculate_distance_of_two_points_of_boxes(first_point, second_point):

    """
    This function calculates a distance l for two input corresponding points of two detected bounding boxes.
    it is assumed that each person is H = 170 cm tall in real scene to map the distances in the image (in pixels) to
    physical distance measures (in meters).
    params:
    first_point: (x, y, h)-tuple, where x,y is the location of a point (center or each of 4 corners of a bounding box)
    and h is the height of the bounding box.
    second_point: same tuple as first_point for the corresponding point of other box
    returns:
    l:  Estimated physical distance (in centimeters) between first_point and second_point.
    """

    # estimate corresponding points distance
    [xc1, yc1, h1] = first_point
    [xc2, yc2, h2] = second_point

    dx = xc2 - xc1
    dy = yc2 - yc1

    lx = dx * 170 * (1 / h1 + 1 / h2) / 2
    ly = dy * 170 * (1 / h1 + 1 / h2) / 2

    l = math.sqrt(lx ** 2 + ly ** 2)

    return l


def get3d(
    x,
    y,
    depth_frame,
):

    # align_to = rs.stream.color
    # align = rs.align(align_to)

    # # Align the depth frame to color frame
    # aligned_frames = align.process(frames)
    # color_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    # # Get aligned frames
    # aligned_depth_frame = aligned_frames.get_depth_frame()

    aligned_depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    dist = meanDepth(depth_frame, x, y)

    udist = depth_frame.get_distance(x, y)

    # depth_pixel = [x, y]
    # # In meters
    point = rs.rs2_deproject_pixel_to_point(aligned_depth_intrin, [x, y], udist)
    # dist_to_center = aligned_depth_frame.get_distance(x, y)

    # The (x,y,z) coordinate system of the camera is accordingly
    # Origin is at the centre of the camera
    # Positive x axis is towards right
    # Positive y axis is towards down
    # Positive z axis is into the 2d xy planedf

    response_dict = {"x": point[0], "y": point[1], "z": point[2]}

    return response_dict


def meanDepth(frame, x, y):
    distList = []
    for i in range(20):

        distList.append(frame.get_distance((x + i), (y + i)))

    distArr = np.array(distList)
    dist = np.mean(distArr)

    return dist


def euclideanDistance(points, min_dist, backup_points, image):

    violate = set()

    for i in range(0, len(points)):

        for j in range(i + 1, len(points)):

            if points[i]["z"] == 0 or points[j]["z"] == 0:
                dist = calculate_distance_of_two_points_of_boxes(
                    backup_points[i], backup_points[j]
                )
            else:
                dist = math.sqrt(
                    (points[i]["x"] - points[j]["x"]) ** 2
                    + (points[i]["y"] - points[j]["y"]) ** 2
                    + (points[i]["z"] - points[j]["z"]) ** 2
                )

            drawLine(
                image,
                (points[i]["x"], points[i]["y"]),
                (points[j]["x"], points[j]["y"]),
                dist,
            )

            if dist < min_dist:

                violate.add(i)
                violate.add(j)

    return violate


def drawLine(image, pt1, pt2, dist):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    cv2.line(image, pt1, pt2, (0, 0, 0))
    cv2.putText(image, str(dist), pt1, font, fontScale, color, thickness)
