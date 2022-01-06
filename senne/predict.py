import os
import numpy as np

import matplotlib.pyplot as plt


class MatrixPredict:
    """ Provide predictions from already trained model in a form of matrices """

    def __init__(self, data_paths: dict, serialized_path: str, device: str = 'cuda'):
        self.data_paths = data_paths
        self.serialized_path = os.path.abspath(serialized_path)
        self.device = device

    def make_prediction(self, vis: bool = False):
        """ Make predictions for matrices placed on desired path """

        # Perform preprocessing

        # Load all networks
        files = os.listdir(self.serialized_path)
        networks = [file for file in files if file.endswith('.pth')]

        for network_name in networks:
            model_path = os.path.join(self.serialized_path, network_name)
