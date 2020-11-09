import threading


class AnonymizeFace(threading.Thread):
    def __init__(self):

        self.face_detector = FaceRecognizer()
    


    def detectFaces(self, color_image):

        cv2.imwrite("full image.jpg", color_image)

        faces_boxes = self.face_detector.predict_faces(color_image)

        for facebox in faces_boxes:
            (dsx, dsy, dex, dey) = facebox
            face = color_image[int(dsy) : int((dey)), int(dsx) : int((dex))]
            cv2.imwrite("face.jpg", face)
            face = self.sladFace(face)
            color_image[int(dsy) : int((dey)), int(dsx) : int((dex))] = face
            cv2.imwrite("returned.jpg", color_image)

        return color_image

    

      def sladFace(self, image, factor=3.0):

        # automatically determine the size of the blurring kernel based
        # on the spatial dimensions of the input image
        (h, w) = image.shape[:2]
        kW = int(w / factor)
        kH = int(h / factor)
        # ensure the width of the kernel is odd
        if kW % 2 == 0:
            kW -= 1
        # ensure the height of the kernel is odd
        if kH % 2 == 0:
            kH -= 1
        # apply a Gaussian blur to the input image using our computed
        # kernel size
        return cv2.GaussianBlur(image, (kW, kH), 0)