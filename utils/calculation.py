def euclideanDistance(points):

    violate = set()

    for i in range(0, len(points)):

        for j in range(i + 1, len(points)):

            dist = math.sqrt(
                (points[i]["x"] - points[j]["x"]) ** 2
                + (points[i]["y"] - points[j]["y"]) ** 2
                + (points[i]["z"] - points[j]["z"]) ** 2
            )

            #print(dist)
            if dist < config.MIN_DISTANCE:

                violate.add(i)
                violate.add(j)

    return violate