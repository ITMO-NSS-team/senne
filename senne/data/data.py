import os
import numpy as np

import torch
import torch.utils.data as data_utils

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from geotiff import GeoTiff

from senne.log import senne_logger


class SenneDataLoader:
    """ Class for data preprocessing and transformations """

    def __init__(self, features_path: str, target_path: str):
        self.features_path = features_path
        self.target_path = target_path

    def explore(self):
        """ Read and visualize geotiff files in the folder. Display all features matrices and label matrix """
        area_folders = os.listdir(self.features_path)
        for current_area in area_folders:
            senne_logger.info(f'Processing area {current_area}')

            # In area folder there are several geotiff files
            area_path = os.path.join(self.features_path, current_area)
            features_array = self._read_geotiff_file(area_path)

            # Read array with labels
            tiff_label_name = ''.join((current_area, '.tif'))
            opened_label_tiff = GeoTiff(os.path.join(self.target_path, tiff_label_name))
            label_matrix = np.array(opened_label_tiff.read())

            # Create plot
            create_matrix_plot(features_array, label_matrix)

    def get_numpy_arrays(self):
        """ Create numpy arrays with source matrices obtained from paths

        :return : numpy array objects for features and target
        """
        area_folders = os.listdir(self.features_path)

        all_features_matrices = []
        all_target_matrices = []
        for current_area in area_folders:
            senne_logger.info(f'Processing area {current_area}')

            # In area folder there are several geotiff files
            area_path = os.path.join(self.features_path, current_area)
            features_array = self._read_geotiff_file(area_path)

            # Read array with labels
            tiff_label_name = ''.join((current_area, '.tif'))
            opened_label_tiff = GeoTiff(
                os.path.join(self.target_path, tiff_label_name))
            label_matrix = np.array(opened_label_tiff.read())

            # Update all features
            all_features_matrices.append(features_array)
            all_target_matrices.append(label_matrix)

        # Convert into pt tensors
        all_features_matrices = np.array(all_features_matrices, dtype='float32')
        all_target_matrices = np.array(all_target_matrices, dtype='float32')

        return all_features_matrices, all_target_matrices

    def get_tensor(self):
        """ Create Pytorch tensor with source matrices obtained from paths

        :return : torch.Tensor objects for features and target
        """
        all_features_matrices, all_target_matrices = self.get_numpy_arrays()
        # Convert numpy arrays into torch tensors
        all_features_matrices = torch.from_numpy(all_features_matrices)
        all_target_matrices = torch.from_numpy(all_target_matrices)

        return all_features_matrices, all_target_matrices

    @staticmethod
    def _read_geotiff_file(area_path: str) -> np.array:
        """ Read and convert geotiff file as numpy array """
        bands_tiff = os.listdir(area_path)
        bands_tiff.sort()

        final_tensor = []
        for tiff_file in bands_tiff:
            tiff_path = os.path.join(area_path, tiff_file)

            opened_tiff = GeoTiff(tiff_path)
            matrix = np.array(opened_tiff.read())

            final_tensor.append(matrix)

        final_tensor = np.array(final_tensor)
        return final_tensor


def create_matrix_plot(features_array: np.array, label_matrix: np.array):
    """ Create plot with several matrices and mask of clouds

    :param features_array: features matrices (four) in one tensor
    :param label_matrix: matrix with labels for each pixel
    """
    fig, axs = plt.subplots(2, 2)
    im_first = axs[0, 0].imshow(features_array[0], cmap='Blues', alpha=1.0)
    # Add label matrix
    axs[0, 0].imshow(label_matrix, cmap='Greys', alpha=0.15)
    divider_first = make_axes_locatable(axs[0, 0])
    cax_first = divider_first.append_axes("right", size="8%", pad=0.2)
    cbar_first = plt.colorbar(im_first, cax=cax_first)
    axs[0, 0].set_title('B02')

    im_second = axs[0, 1].imshow(features_array[1], cmap='Greens', alpha=1.0)
    # Add label matrix
    axs[0, 1].imshow(label_matrix, cmap='Greys', alpha=0.15)
    divider_second = make_axes_locatable(axs[0, 1])
    cax_second = divider_second.append_axes("right", size="8%", pad=0.2)
    cbar_second = plt.colorbar(im_second, cax=cax_second)
    axs[0, 1].set_title('B03')

    im_third = axs[1, 0].imshow(features_array[2], cmap='Oranges', alpha=1.0)
    # Add label matrix
    axs[1, 0].imshow(label_matrix, cmap='Greys', alpha=0.15)
    divider_third = make_axes_locatable(axs[1, 0])
    cax_third = divider_third.append_axes("right", size="8%", pad=0.2)
    cbar_third = plt.colorbar(im_third, cax=cax_third)
    axs[1, 0].set_title('B04')

    im_fourth = axs[1, 1].imshow(features_array[3], cmap='Reds', alpha=1.0)
    # Add label matrix
    axs[1, 1].imshow(label_matrix, cmap='Greys', alpha=0.15)
    divider_fourth = make_axes_locatable(axs[1, 1])
    cax_fourth = divider_fourth.append_axes("right", size="8%", pad=0.2)
    cbar_fourth = plt.colorbar(im_fourth, cax=cax_fourth)
    axs[1, 1].set_title('B08')

    plt.tight_layout()
    plt.show()


def train_test(x_train: torch.tensor, y_train: torch.tensor,
               train_size: float = 0.8):
    """ Method for train test split

    :param x_train: pytorch tensor with features
    :param y_train: pytorch tensor with labels
    :param train_size: value from 0.1 to 0.9
    """
    if train_size < 0.1 or train_size > 0.99:
        raise ValueError('train_size value must be value between 0.1 and 0.99')
    dataset = data_utils.TensorDataset(x_train, y_train)
    train_ratio = round(len(dataset) * train_size)
    test_ratio = len(dataset) - train_ratio
    train, test = torch.utils.data.random_split(dataset,
                                                [train_ratio, test_ratio])

    train_features, train_target = train.dataset[train.indices]
    test_features, test_target = test.dataset[test.indices]
    train_dataset = data_utils.TensorDataset(train_features, train_target)
    test_dataset = data_utils.TensorDataset(test_features, test_target)
    return train_dataset, test_dataset
