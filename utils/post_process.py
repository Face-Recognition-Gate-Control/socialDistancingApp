import pyrealsense2 as rs
import math
import detect.config_caffe as config
import cv2
import numpy as np


def drawBox(image, predicitons, min_dist):
    violation = set()

    overlay = image.copy()
    output = image.copy()
    alpha = 0.35

    if len(predicitons[1]) >= 2:

        violation = euclideanDistance(predicitons[1], min_dist)

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
    color_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    # # Get aligned frames
    # aligned_depth_frame = aligned_frames.get_depth_frame()

    # aligned_depth_intrin = (
    #     aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    # )

    dist = meanDepth(depth_frame, x, y)

    udist = depth_frame.get_distance(x, y)

    # depth_pixel = [x, y]
    # # In meters
    point = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], udist)
    # dist_to_center = aligned_depth_frame.get_distance(x, y)

    # The (x,y,z) coordinate system of the camera is accordingly
    # Origin is at the centre of the camera
    # Positive x axis is towards right
    # Positive y axis is towards down
    # Positive z axis is into the 2d xy plane

    response_dict = {"x": point[0], "y": point[1], "z": point[2]}

    return response_dict


def meanDepth(frame, x, y):
    distList = []
    for i in range(20):

        distList.append(frame.get_distance((x + i), (y + i)))

    distArr = np.array(distList)
    dist = np.mean(distArr)

    return dist


def euclideanDistance(points, min_dist):

    violate = set()

    for i in range(0, len(points)):

        for j in range(i + 1, len(points)):

            if(points[i]['z']==0 orpoints[j]['z']==0 ):
                print('hei')
            dist = math.sqrt(
                (points[i]["x"] - points[j]["x"]) ** 2
                + (points[i]["y"] - points[j]["y"]) ** 2
                + (points[i]["z"] - points[j]["z"]) ** 2
            )

            if dist < min_dist:

                violate.add(i)
                violate.add(j)

    return violate