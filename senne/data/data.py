import json
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split

from geotiff import GeoTiff
from typing import List, Optional

from senne.data.preprocessing import apply_normalization
from senne.log import senne_logger

TRAIN_SIZE = 0.7


class DataProcessor:
    """ Class for data preprocessing and transformations """

    def __init__(self, features_path: str, target_path: str):
        self.features_path = os.path.abspath(features_path)
        self.target_path = os.path.abspath(target_path)

        self.train_samples_file_name = 'train.csv'
        self.test_samples_file_name = 'test.csv'
        self.min_max_file_name = 'boundaries.json'

    def explore(self):
        """
        Read and visualize geotiff files in the folder.
        Display all features matrices and label matrix
        """
        chip_folders = os.listdir(self.features_path)
        for current_chip in chip_folders:
            senne_logger.info(f'Processing chip {current_chip}')

            # In chip folder there are several geotiff files
            chip_path = os.path.join(self.features_path, current_chip)
            features_array = self.read_geotiff_file(chip_path)

            # Read array with labels
            tiff_label_name = ''.join((current_chip, '.tif'))
            opened_label_tiff = GeoTiff(os.path.join(self.target_path,
                                                     tiff_label_name))
            label_matrix = np.array(opened_label_tiff.read())

            # Create plot
            create_matrix_plot(features_array, label_matrix)

    def check_bands_number(self, correct_band_number: int):
        """ Check if features contain desired number of bands """
        chip_folders = os.listdir(self.features_path)
        for current_chip in chip_folders:
            chip_path = os.path.join(self.features_path, current_chip)
            train_files = os.listdir(chip_path)

            if len(train_files) != correct_band_number:
                print(f'Chip with path {chip_path} is incorrect')

    def collect_sample_info(self, serialized_path: str):
        """ Explore training sample - collect dataframe with all samples.
        Create json file with boundaries for normalization for each band

        :param serialized_path: path to folder for serialization
        """
        # Check if files already exist in the path or not
        if self._is_sample_info_already_collected(serialized_path):
            return None

        chip_folders = os.listdir(self.features_path)
        features_paths = []
        target_paths = []
        cloud_ratio = []

        features_info = {}
        boundaries = {}
        for chip_id, current_chip in enumerate(chip_folders):
            senne_logger.info(f'Processing chip {current_chip}')

            # In chip folder there are several geotiff files
            chip_path = os.path.join(self.features_path, current_chip)
            features_array = self.read_geotiff_file(chip_path)

            if chip_id == 0:
                # Store info about number of bands into separate field
                features_info['in_channels'] = len(features_array)

            # Read array with labels
            tiff_label_name = ''.join((current_chip, '.tif'))
            target_path = os.path.join(self.target_path, tiff_label_name)
            if os.path.isfile(target_path) is False:
                continue

            self.define_normalization_boundaries(features_array, boundaries)

            features_paths.append(os.path.abspath(chip_path))
            target_paths.append(os.path.abspath(target_path))

            opened_label_tiff = GeoTiff(target_path)
            label_matrix = np.array(opened_label_tiff.read())
            n_rows, n_cols = label_matrix.shape
            cloud_percent = len(np.argwhere(label_matrix == 1)) / (n_rows * n_cols)
            cloud_ratio.append(round(cloud_percent, 3))

        # Store dataframe with paths
        df_samples = pd.DataFrame({'feature': features_paths, 'target': target_paths,
                                   'cloud_ratio': cloud_ratio})
        # Divide into train and test and save into csv files
        train_df, test_df = train_test_split(df_samples, train_size=TRAIN_SIZE)
        train_df.to_csv(os.path.join(serialized_path, self.train_samples_file_name),
                        index=False)
        test_df.to_csv(os.path.join(serialized_path, self.test_samples_file_name),
                       index=False)

        # Save boundaries
        json_path = os.path.join(serialized_path, self.min_max_file_name)
        features_info['info'] = boundaries
        with open(json_path, 'w') as f:
            json.dump(features_info, f)

    def _is_sample_info_already_collected(self, serialized_path):
        serialized_files = os.listdir(serialized_path)
        is_boundary_collected = self.min_max_file_name in serialized_files
        is_train_collected = self.train_samples_file_name in serialized_files
        is_test_collected = self.test_samples_file_name in serialized_files
        if is_boundary_collected and is_train_collected and is_test_collected:
            return True
        else:
            return False

    @staticmethod
    def define_normalization_boundaries(features_array: np.array, boundaries: dict):
        """
        Define and update (if necessary) information about features matrices (bands).

        :param features_array: array with dimensions n, rows, columns where n
        is the number of bands
        :param boundaries: dictionary with bounds (min and max) per each band
        """
        for band_number, band_matrix in enumerate(features_array):
            min_band_value = np.min(band_matrix)
            max_band_value = np.max(band_matrix)

            current_band_info = boundaries.get(band_number)
            if current_band_info is None:
                # Create initial min max dictionary info
                boundaries[band_number] = {'min': float(min_band_value),
                                           'max': float(max_band_value)}
            else:
                # Boundaries are already exist
                if min_band_value < current_band_info['min']:
                    current_band_info['min'] = float(min_band_value)
                if max_band_value > current_band_info['max']:
                    current_band_info['max'] = float(max_band_value)

    @staticmethod
    def read_geotiff_file(area_path: str) -> np.array:
        """ Read and convert geotiff file as numpy array """
        bands_tiff = os.listdir(area_path)
        bands_tiff.sort()

        final_tensor = []
        for tiff_file in bands_tiff:
            tiff_path = os.path.join(area_path, tiff_file)

            opened_tiff = GeoTiff(tiff_path)
            matrix = np.array(opened_tiff.read())

            final_tensor.append(matrix)

        final_tensor = np.array(final_tensor, dtype='float32')
        return final_tensor

    def in_channels(self, serialized_path: str):
        """ Return number of channels for dataset """
        file_path = os.path.join(serialized_path, self.min_max_file_name)
        with open(file_path) as json_file:
            preprocess_info = json.load(json_file)

        return preprocess_info['in_channels']


class SenneDataset(data_utils.Dataset):
    """ Torch-like Dataset for processing images into SENNE library """

    def __init__(self, serialized_folder: str, dataframe_with_paths: pd.DataFrame,
                 transforms: Optional[str] = None, for_train: bool = True):
        self.serialized_folder = serialized_folder
        self.dataframe_with_paths = dataframe_with_paths
        self.transforms = transforms
        self.for_train = for_train

        self.perform_dataset_preprocessing(transforms)

        # Load information about boundaries from json file
        file_path = os.path.join(serialized_folder, 'boundaries.json')
        with open(file_path) as json_file:
            self.boundaries_info = json.load(json_file)

    def __len__(self):
        return len(self.dataframe_with_paths)

    def __getitem__(self, index: int):
        # Firstly load images
        row_object = self.dataframe_with_paths.loc[index]
        feature_path = row_object['feature']
        target_path = row_object['target']

        # Convert into numpy arrays
        feature_array = DataProcessor.read_geotiff_file(feature_path)
        opened_label_tiff = GeoTiff(target_path)
        label_matrix = np.array(opened_label_tiff.read(), dtype='float32')
        label_matrix = np.expand_dims(label_matrix, 0)
        self.perform_matrix_preprocessing(feature_array, label_matrix)

        # Convert into tensors
        feature_array = torch.from_numpy(feature_array)
        label_matrix = torch.from_numpy(label_matrix)
        return feature_array, label_matrix

    def perform_dataset_preprocessing(self, transforms: str):
        # There are no dataset preprocessing procedures yet
        self.dataframe_with_paths = self.dataframe_with_paths.reset_index()
        if self.for_train is False:
            # There is no need to do transformations
            return None

    def perform_matrix_preprocessing(self, feature_array: np.array, label_matrix: np.array):
        # Obligatory preprocessing is applied
        feature_array = apply_normalization(feature_array, self.boundaries_info)
        if self.for_train is False:
            return feature_array, label_matrix

        return feature_array, label_matrix


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
