
# calcualate the distance from all the coordinates(x,y,z) from detected personns
def detectHog(frame, HOGCV):

    frame = cv2.resize(frame, (300, 300))
    print(frame.shape)
    start = datetime.datetime.now()

    # Factor for scale to original size of frame
    heightFactor = frame.shape[0] / 300.0
    widthFactor = frame.shape[1] / 300.0

    (rects, weights) = HOGCV.detectMultiScale(
        frame, winStride=(4, 4), padding=(128, 128), scale=1.05
    )

    print(rects)
    rects = np.array(
        [
            [
                int(x * widthFactor),
                int(y * heightFactor),
                int(w * widthFactor),
                int(h * heightFactor),
            ]
            for (x, y, w, h) in rects
        ]
    )
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    print("people", len(rects))

    # print(
    #     "[INFO] detection took: {}s".format(
    #         (datetime.datetime.now() - start).total_seconds()
    #     )
    # )

    # for box, weight in zip(bounding_box_cordinates, weights):
    #     print(box, weight)

    return result