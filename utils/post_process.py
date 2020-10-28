import pyrealsense2 as rs
import math
import detect.config_caffe as config
import cv2


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


def get3d(x, y, depth_frame):

    # align_to = rs.stream.color
    # align = rs.align(align_to)

    # # Align the depth frame to color frame
    # aligned_frames = align.process(frames)

    # # Get aligned frames
    # aligned_depth_frame = aligned_frames.get_depth_frame()

    # aligned_depth_intrin = (
    #     aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    # )

    # depth_pixel = [x, y]
    # # In meters
    # dist_to_center = aligned_depth_frame.get_distance(x, y)
    # pose = rs.rs2_deproject_pixel_to_point(
    #     aligned_depth_intrin, depth_pixel, dist_to_center
    # )

    z = depth_frame.get_distance(x, y)

    print(z)

    # The (x,y,z) coordinate system of the camera is accordingly
    # Origin is at the centre of the camera
    # Positive x axis is towards right
    # Positive y axis is towards down
    # Positive z axis is into the 2d xy plane

    response_dict = {"x": x, "y": y, "z": z}

    return response_dict


def euclideanDistance(points, min_dist):

    violate = set()

    for i in range(0, len(points)):

        for j in range(i + 1, len(points)):

            dist = math.sqrt(
                (points[i]["x"] - points[j]["x"]) ** 2
                + (points[i]["y"] - points[j]["y"]) ** 2
                + (points[i]["z"] - points[j]["z"]) ** 2
            )

            if dist < min_dist:

                violate.add(i)
                violate.add(j)

    return violate