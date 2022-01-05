import os
import cv2 as cv
import numpy as np

from senne.data.data import SenneDataLoader
from senne.repository.presets import *


class Ensembler:
    """
    Class for ensembing several neural networks for semantic segmentation.
    For now, only binary segmentation is supported.

    Description of presets and preprocessing strategies
    networks_by_preset: stack of neural networks to use
        * two_simple - create two light neural networks
        * three_simple - create three simple neural networks
        * four_simple - create four simple neural networks
        * two_advanced - create two advanced neural networks and launch advanced preprocessing

    preprocessing_by_preset: which strategy for data preprocessing should be applied
        * default - apply only scaling and stacking
        * augmentation - apply augmentation for images to extend train sample size
        * filter_min - filter images and stay only images with low cloud ratio (less than 20%)
        * filter_max - filter images and stay only images with high cloud ratio (more than 80%)
        * remote_cloud - use advanced method from remote sensing to calculate special indices

    Sign ' | ' applied for composite preprocessing pipelines with several stages.
    Possible combinations are ''
    """
    networks_by_preset = {'two_simple': create_two_simple_networks,
                          'three_simple': create_three_simple_networks,
                          'four_simple': create_four_simple_networks,
                          'two_advanced': create_two_advanced_networks}

    preprocessing_by_preset = {'two_simple': ['default', 'default'],
                               'three_simple': ['default', 'augmentation',
                                                'filter'],
                               'four_simple': ['default', 'augmentation',
                                               'filter_min', 'filter_max'],
                               'two_advanced': ['filter_min | augmentation',
                                                'filter_max | remote_cloud | augmentation']}

    def __init__(self, path: str, device: str = 'cuda'):
        self.path = os.path.abspath(path)
        # Create folder if necessary
        self._create_folder(self.path)

        self.device = device

        # Classes for data preprocessing
        self.data_loader = None
        self.remote_preprocessor = None
        self.image_preprocessor = None

    def initialise_nn_models(self, preset: str) -> list:
        """ Create several neural networks for experiments """
        raise NotImplementedError()

    def train_neural_networks(self, data_paths: dict, preset: str):
        """ Train several neural networks for image segmentation

        :param data_paths:
        :param preset: define which neural networks to use for ensembling
        """
        # Load data and convert it into pt files
        self.data_loader = SenneDataLoader(features_path=data_paths['features_path'],
                                           target_path=data_paths['target_path'])
        features_tensor, target_tensor = self.data_loader.get_numpy_arrays()

        # Launch data preparation with preprocessing
        preprocessing_names = self.preprocessing_by_preset[preset]
        self.apply_preprocessing(features_tensor, target_tensor, preprocessing_names)

        nn_models = self.initialise_nn_models(preset)

    def apply_preprocessing(self, features_tensor: np.array, target_tensor: np.array,
                            preprocessing_names: list):
        """ Apply preprocessing on PyTorch tensors

        :param features_tensor: PyTorch tensor with source features matrices
        :param target_tensor: PyTorch tensor with source target matrices
        :param preprocessing_names: list with names of preprocessing strategies
        """
        for i, preprocessing_name in enumerate(preprocessing_names):
            copied_features = np.copy(features_tensor)
            copied_target = np.copy(target_tensor)

            # TODO implement different preprocessing strategies
            prep_folder = os.path.join(self.path, ''.join(('preprocessing_', str(i))))
            self._create_folder(prep_folder)

            # Apply normalization - default preprocessing for every strategy
            copied_features, copied_target = normalize(copied_features, copied_target)

            # Save tensors in pt files
            copied_features = torch.from_numpy(copied_features)
            copied_target = torch.from_numpy(copied_target)
            torch.save(copied_features, os.path.join(prep_folder, 'features.pt'))
            torch.save(copied_target, os.path.join(prep_folder, 'target.pt'))

    @staticmethod
    def _create_folder(path):
        if os.path.isdir(path) is False:
            os.makedirs(path)


def normalize(features_tensor: np.array, target_tensor: np.array):
    """ Perform normalization procedure for numpy arrays """
    n_objects, n_bands, n_rows, n_columns = features_tensor.shape
    for band_id in range(n_bands):
        band_tensor = features_tensor[:, band_id, :, :]

        min_value = float(np.min(band_tensor))
        max_value = float(np.max(band_tensor))
        features_tensor[:, band_id, :, :] = cv.normalize(band_tensor, min_value, max_value)

    return features_tensor, target_tensor
