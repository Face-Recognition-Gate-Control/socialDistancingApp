# from .vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
# from .vision.ssd.config.fd_config import define_img_size
from .vision.ssd.mb_tiny_RFB_fd import (
    create_Mb_Tiny_RFB_fd,
    create_Mb_Tiny_RFB_fd_predictor,
)
from .vision.ssd.config.fd_config import define_img_size
import torch


PATH_TO_FACE_DETECTION_MODEL = "./models/RFB-640/face_model.pth"


class FaceRecognizer:
    """
    This class represents the a light-weight face detection model
    designed for edge computing devices
    """

    INPUT_IMAGE_SIZE = 1280  # default cv2-size
    NUMB_OF_RESULTS = 1  # keep 'x' results
    PROB_THRESHOLD = 0.7  # sens 0 - 1 (1=highest)
    # only consider the candidates with the highest scores.
    CANDIDATE_SIZE = 500

    def __init__(self, default_size=640):
        """Initializes an instance of the face-prediction model with
        given set of configurations based upon the network structure and model type

        Args:
            model_path ([str]): [path to stored model]
            default_size (int, optional): [normal expected input size to model]. Defaults to 640.
        """
        self.model_path = PATH_TO_FACE_DETECTION_MODEL
        self.torch_device = self._create_and_get_device()
        self._set_network_image_size(image_size=default_size)
        self.net = self._get_network_structure()
        self.predictor = self._get_predictor()

    def _get_network_structure(self):
        """Generates a neural network layer structure for the model,
        based upon number of classes, test-type and its device-options

        Returns:
            [nn.Module]: [neural network layer structure]
        """
        # TODO: move to somewhere else
        is_testing = True  # evaluation mode
        num_classes = 2  # background and face

        network = create_Mb_Tiny_RFB_fd(
            num_classes=num_classes, is_test=is_testing, device=self.torch_device
        )
        network.load(self.model_path)
        return network

    def _get_predictor(self):
        """Generates a Predictor with a given network structure, used to
        setup the configurations for the face prediction

        Returns:
            [Predictor]: [a wrapper for the buit nn.Module structure]
        """
        return create_Mb_Tiny_RFB_fd_predictor(
            net=self.net,
            candidate_size=FaceRecognizer.CANDIDATE_SIZE,
            device=self.torch_device,
        )

    def predict_faces(self, frame):
        """receives a frame from the webcam,
        and attempts to predict the position of the face if detected

        Args:
            frame ([numpy.ndarray]): [the whole camera frame]

        Returns:
            [Tensor]: [list of prediction coords]
        """
        boxes, labels, probs = self.predictor.predict(
            image=frame,
            top_k=FaceRecognizer.NUMB_OF_RESULTS,
            prob_threshold=FaceRecognizer.PROB_THRESHOLD,
        )
        return boxes

    @staticmethod
    def _create_and_get_device():
        """validates the computers/systems capabilities,
                and loads the device on which a torch.Tensor is or will be allocated.

        Returns:
            [str]: [the name of allocated device]
        """
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    @staticmethod
    def _set_network_image_size(image_size):
        """generates the global expected image input size to the nn.structre

        Args:
            image_size ([int]): [image x-length]
        """
        define_img_size(image_size)


if __name__ == "__main__":

    print(create_Mb_Tiny_RFB_fd_predictor)