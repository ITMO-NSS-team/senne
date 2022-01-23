import json
import os
import pickle
from copy import copy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import torch
from geotiff import GeoTiff
from mpl_toolkits.axes_grid1 import make_axes_locatable

from senne.data.data import DataProcessor
from senne.data.preprocessing import apply_normalization
from tifffile import imsave


class MatrixPredict:
    """
    Provide predictions from already trained model in a form of matrices.
    To avoid problems with memory storage all processing operations perform
    iteratively on each tile
    """

    def __init__(self, data_paths: dict, serialized_path: str, device: str = 'cuda',
                 final_model: str = 'logit'):
        self.data_paths = data_paths
        self.serialized_path = os.path.abspath(serialized_path)
        self.device = device
        self.final_model = final_model

        # Target path may be defined as None
        self.data_loader = DataProcessor(features_path=data_paths['features_path'],
                                         target_path=data_paths.get('target_path'))
        self.expected_n_rows = None
        self.expected_n_cols = None
        self.expected_n_objects = 0

    def make_prediction(self, save_folder: str, vis: bool = False, take_first_images: int = None):
        """ Make predictions for matrices placed on desired path

        :param save_folder: path to the folder where to save geotiff files
        :param vis: is there a need to make visualizations
        :param take_first_images: how many images needed to be processed
        """
        if vis and self.data_paths.get('target_path') is None:
            raise ValueError(f'For visualisation actual target needed!')

        # Create folder
        save_folder = os.path.abspath(save_folder)
        if os.path.isdir(save_folder) is False:
            os.makedirs(save_folder)

        preprocess_json = os.path.join(self.serialized_path, 'preprocessing.json')
        with open(preprocess_json) as json_file:
            preprocess_info = json.load(json_file)

        # Define paths to networks
        files = os.listdir(self.serialized_path)
        networks = [file for file in files if file.endswith('.pth')]
        # Load models into dictionary
        nn_models = self._load_neural_networks(networks)

        # Find all files (areas) to process
        geotiff_files = os.listdir(self.data_paths['features_path'])

        if take_first_images is not None:
            geotiff_files = geotiff_files[: take_first_images]

        for file_id, geotiff_file in enumerate(geotiff_files):
            ################################
            # Load source geotiff matrices #
            ################################
            try:
                # Read current geotiff file
                area_path = os.path.join(self.data_paths['features_path'], geotiff_file)
                features_tensor = self.data_loader.read_geotiff_file(area_path)

                if self.data_paths.get('target_path') is not None:
                    target_name = ''.join((geotiff_file, '.tif'))
                    target_path = os.path.join(self.data_paths['target_path'], target_name)
                    opened_label_tiff = GeoTiff(target_path)
                    target_matrix = np.array(opened_label_tiff.read())
                else:
                    target_matrix = None

            except FileNotFoundError as ex:
                print(f'{ex.__str__()}')
                # Pass this iteration
                continue

            # Initialize final report dictionary
            features_df = {}
            self.expected_n_objects += 1
            for i, network_name in enumerate(networks):
                features_copied = np.copy(features_tensor)

                current_model_info = preprocess_info[network_name.split('.')[0]]
                features_copied = prepare_data_for_predict(features_copied,
                                                           current_model_info)

                current_model = nn_models[network_name]
                # Get neural networks output
                pr_mask = current_model.predict(features_copied.to(self.device))
                pr_mask = pr_mask.squeeze().cpu().numpy()

                # Update information about predictions
                features_df[network_name] = np.ravel(pr_mask)

                if i == len(networks) - 1:
                    # Last neural network give prediction
                    all_rows, all_cols = create_row_column_features_for_predict(pr_mask)
                    features_df['row_ids'] = all_rows
                    features_df['col_ids'] = all_cols

                    if target_matrix is not None:
                        features_df['target'] = np.ravel(target_matrix)

                if file_id == 0:
                    self.expected_n_rows, self.expected_n_cols = pr_mask.shape

            features_df = pd.DataFrame(features_df)
            if self.final_model == 'automl':
                # Use pipeline to ensemble forecast
                predicted = self.pipeline_predict(features_df, network_names=copy(networks))
            else:
                predicted = self.final_model_predict(features_df, network_names=copy(networks))

            predicted_matrix = self.transform_one_dim_into_matrices(predicted)

            # Save matrix as geotiff file
            file_path = os.path.join(save_folder, ''.join((geotiff_file, '.tif')))
            imsave(file_path, np.array(predicted_matrix, dtype="uint8"))

            if vis:
                self.display_predictions(predicted, features_df)

    def final_model_predict(self, features_df, network_names):
        network_names.sort()
        network_names.extend(['row_ids', 'col_ids'])

        # Search for pkl file
        path = os.path.join(self.serialized_path, 'final_model.pkl')
        with open(path, "rb") as f:
            final_model = pickle.load(f)

        labels_output = final_model.predict(np.array(features_df[network_names]))
        return labels_output

    def pipeline_predict(self, features_df, network_names):
        """
        Make a final prediction using a pipeline

        :param features_df: pandas DataFrame with features
        :param network_names: list with names of CNN models
        :return: labels from the final model
        """
        from fedot.core.pipelines.pipeline import Pipeline

        network_names.sort()
        network_names.extend(['row_ids', 'col_ids'])

        input_data = prepare_input_data(np.array(features_df[network_names]))

        # Load pipeline
        pipeline_path = os.path.join(self.serialized_path, 'final_model', 'final_model.json')
        pipeline = Pipeline()
        pipeline.load(pipeline_path)

        labels_output = pipeline.predict(input_data, output_mode='labels').predict

        return labels_output

    def transform_one_dim_into_matrices(self, prediction: np.array):
        """ Transform one-dimensional prediction into matrix """
        return prediction.reshape(self.expected_n_rows, self.expected_n_cols)

    def display_predictions(self, pipeline_predicted: np.array, features_df):
        """ Prepare plot with several neural networks and real target """
        pipeline_matrix = self.transform_one_dim_into_matrices(pipeline_predicted)
        target_matrix = self.transform_one_dim_into_matrices(np.array(features_df['target']))
        nn_first_matrix = self.transform_one_dim_into_matrices(np.array(features_df['network_0.pth']))
        nn_second_matrix = self.transform_one_dim_into_matrices(np.array(features_df['network_1.pth']))

        for tensor in [pipeline_matrix, target_matrix, nn_first_matrix, nn_second_matrix]:
            tensor[tensor >= 0.5] = 1
            tensor[tensor < 0.5] = 0

        fig, axs = plt.subplots(2, 2)
        im_first = axs[0, 0].imshow(nn_first_matrix, cmap='Blues', alpha=1.0,
                                    vmin=0, vmax=1)
        divider_first = make_axes_locatable(axs[0, 0])
        cax_first = divider_first.append_axes("right", size="8%", pad=0.2)
        cbar_first = plt.colorbar(im_first, cax=cax_first)
        axs[0, 0].set_title('Network first predict')

        im_second = axs[0, 1].imshow(nn_second_matrix, cmap='Blues', alpha=1.0,
                                     vmin=0, vmax=1)
        divider_second = make_axes_locatable(axs[0, 1])
        cax_second = divider_second.append_axes("right", size="8%", pad=0.2)
        cbar_second = plt.colorbar(im_second, cax=cax_second)
        axs[0, 1].set_title('Network second predict')

        im_third = axs[1, 0].imshow(pipeline_matrix, cmap='Blues', alpha=1.0,
                                    vmin=0, vmax=1)
        divider_third = make_axes_locatable(axs[1, 0])
        cax_third = divider_third.append_axes("right", size="8%", pad=0.2)
        cbar_third = plt.colorbar(im_third, cax=cax_third)
        axs[1, 0].set_title('Ensemble prediction')

        im_fourth = axs[1, 1].imshow(target_matrix, cmap='Blues', alpha=1.0,
                                     vmin=0, vmax=1)
        divider_fourth = make_axes_locatable(axs[1, 1])
        cax_fourth = divider_fourth.append_axes("right", size="8%", pad=0.2)
        cbar_fourth = plt.colorbar(im_fourth, cax=cax_fourth)
        axs[1, 1].set_title('Actual values')

        plt.tight_layout()
        plt.show()

    def _load_neural_networks(self, networks_names: list) -> dict:
        """ Load PyTorch neural networks from serialized path using names """
        models = {}
        for network_name in networks_names:
            model_path = os.path.join(self.serialized_path, network_name)
            nn_model = torch.load(model_path)
            models.update({network_name: nn_model})

        return models


def prepare_data_for_predict(features_tensor: np.array, preprocess_info: dict):
    """ Prepare pytorch tensors """
    # Normalize data
    features_tensor = apply_normalization(np.expand_dims(features_tensor, axis=0),
                                          preprocess_info)
    features_tensor = torch.from_numpy(features_tensor)
    return features_tensor


def create_row_column_features_for_predict(predictions: np.array):
    """ Create two columns (features) with indices

    :param predictions: output from neural network in a form of multidimensional array
    """
    n_rows, n_cols = predictions.shape
    # Column indices
    row_ids = np.arange(0, n_rows).reshape((-1, 1))
    row_ids = np.repeat(row_ids, n_cols, axis=1)
    all_row_matrix = np.ravel(row_ids)

    # Column indices
    col_ids = np.arange(0, n_cols).reshape((1, -1))
    col_ids = np.repeat(col_ids, n_rows, axis=0)
    all_col_matrix = np.ravel(col_ids)

    return all_row_matrix, all_col_matrix


def prepare_input_data(features):
    """
    For FEDOT to run on new data not through API, you need to put the array
    in a special data class (InputData)
    """
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    task = Task(TaskTypesEnum.classification)
    features_input = InputData(idx=np.arange(len(features)),
                               features=features, target=None,
                               task=task, data_type=DataTypesEnum.table)

    return features_input
