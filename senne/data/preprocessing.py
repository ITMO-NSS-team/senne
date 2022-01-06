import cv2 as cv
import numpy as np

from senne.log import senne_logger


class RemoteSensingPreprocessor:
    """ Class for applying data preprocessing for remote sensing data during ensembling """

    def __init__(self):
        pass

    def smooth(self):
        """ Apply gaussian smoothing for images """
        raise NotImplementedError()


class ImagePreprocessor:
    """
    Class for preparing images using augmentation methods and other classical
    algorithms for image processing
    """

    def __init__(self):
        pass

    def smooth(self):
        """ Apply gaussian smoothing for images """
        raise NotImplementedError()


def normalize(features_tensor: np.array, target_tensor: np.array):
    """ Perform normalization procedure for numpy arrays """
    n_objects, n_bands, n_rows, n_columns = features_tensor.shape
    band_boundaries = {}
    for band_id in range(n_bands):
        band_tensor = features_tensor[:, band_id, :, :]

        min_value = float(np.min(band_tensor))
        max_value = float(np.max(band_tensor))
        features_tensor[:, band_id, :, :] = cv.normalize(band_tensor, min_value, max_value)

        # Save information about transformations
        band_boundaries.update({band_id: {'min': min_value, 'max': max_value}})

    return features_tensor, target_tensor, band_boundaries


def apply_normalization(features_tensor: np.array, preprocessing_info: dict):
    """ Perform normalization procedure for new unseen data """
    n_objects, n_bands, n_rows, n_columns = features_tensor.shape

    for band_id in range(n_bands):
        band_tensor = features_tensor[:, band_id, :, :]
        min_value = preprocessing_info['info'][str(band_id)]['min']
        max_value = preprocessing_info['info'][str(band_id)]['max']

        features_tensor[:, band_id, :, :] = cv.normalize(band_tensor, min_value, max_value)

    return features_tensor
