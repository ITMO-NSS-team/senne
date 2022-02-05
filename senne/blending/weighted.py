import os
import pickle

import pandas as pd
import numpy as np
import torch
import segmentation_models_pytorch as smp
from functools import partial

from hyperopt import hp, fmin, tpe, space_eval

from senne.blending.blending import AbstractEnsemble
from senne.log import senne_logger


class WeightedEnsemble(AbstractEnsemble):
    """ Class for ensemble forecasts from several neural networks """

    def __init__(self, boundaries_info: dict, networks_info: dict, path: str,
                 device: str, metadata_path: str = None, for_predict: bool = False):
        super().__init__(boundaries_info, networks_info, path, device, metadata_path, for_predict)

        # Initialize parameters (weights) and search space for training
        self.parameters = {}
        self.search_space = {}

    def fit(self, iterations: int = 10):
        """ Perform weighted model training """
        df_paths = pd.read_csv(os.path.join(self.path, 'test.csv'))
        n_objects = len(df_paths)

        self._init_networks_dataset(df_paths)
        self._init_networks_models()
        self._init_parameters_and_search_space()

        senne_logger.info('Optimization process has been started')
        self.__optimize_parameters(n_objects, iterations)

        # Save current model
        self.for_predict = True
        save_path = os.path.join(self.path, 'final_model.pkl')
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    def predict(self, chip_arr: np.array, chip_name: str = None, use_network: int = None):
        """ Perform predict for ensemble with several neural networks

        :param chip_arr: numpy array with features matrices
        :param chip_name: name of chip to prepare forecast for
        :param use_network: id of neural network to use for forecasting. If None -
        use all neural networks
        """
        # Take month from datetime and store in into datetime column
        if chip_name is not None:
            self.metadata['datetime'] = self.metadata['datetime'].dt.month
        network_files = [file for file in os.listdir(self.path) if '.pth' in file]
        network_files.sort()

        if use_network is None:
            pr_mask = self.predict_based_on_all_networks(network_files, chip_arr, chip_name)
        else:
            # Use single neural network
            pr_mask = self.predict_based_on_single_networks(network_files, chip_arr, use_network)
        return pr_mask

    def predict_based_on_all_networks(self, network_files: list, chip_arr: np.array,
                                      chip_name: str = None) -> np.array:
        """ Make forecast using all neural networks """
        predicted_masks = []
        weights = []
        for network_id, network_file in enumerate(network_files):
            features_tensor = np.copy(chip_arr)
            # Perform preprocessing for chip
            features_tensor = self._preprocess_field(network_file,
                                                     features_tensor)

            # Choose appropriate neural network
            current_neural_network = self.nn_models[network_id]
            features_tensor = torch.from_numpy(features_tensor).to(self.device).unsqueeze(0)
            pr_mask = current_neural_network.predict(features_tensor)
            # Into numpy array
            pr_mask = pr_mask.squeeze().cpu().numpy()

            # Binarization
            if chip_name is None:
                threshold = self.parameters[f'th_{network_id}']
                pr_mask[pr_mask >= threshold] = 1
                pr_mask[pr_mask < threshold] = 0

                predicted_masks.append(pr_mask)
                weights.append(self.parameters[f'weight_{network_id}'])
            else:
                # Particular weights for each region
                chip_metadata = self.metadata[
                    self.metadata['chip_id'] == chip_name]
                current_time_location = str(chip_metadata['datetime'].values[0])

                threshold = self.parameters[
                    f'{current_time_location}_th_{network_id}']
                pr_mask[pr_mask >= threshold] = 1
                pr_mask[pr_mask < threshold] = 0

                predicted_masks.append(pr_mask)
                weights.append(self.parameters[
                                   f'{current_time_location}_weight_{network_id}'])

        predicted_masks = np.array(predicted_masks, dtype=float)
        pr_mask = np.average(predicted_masks, axis=0, weights=weights)

        pr_mask[pr_mask >= 0.5] = 1
        pr_mask[pr_mask < 0.5] = 0

        pr_mask = pr_mask.astype(np.uint8)

        return pr_mask

    def predict_based_on_single_networks(self, network_files: list, chip_arr: np.array,
                                         network_id: int) -> np.array:
        network_file = network_files[network_id]
        # Perform preprocessing for chip
        chip_arr = self._preprocess_field(network_file, chip_arr)

        # Choose appropriate neural network
        current_neural_network = self.nn_models[network_id]
        chip_arr = torch.from_numpy(chip_arr).to(self.device).unsqueeze(0)
        pr_mask = current_neural_network.predict(chip_arr)
        # Into numpy array
        pr_mask = pr_mask.squeeze().cpu().numpy()

        # Binarization
        pr_mask[pr_mask >= 0.5] = 1
        pr_mask[pr_mask < 0.5] = 0

        pr_mask = pr_mask.astype(np.uint8)

        return pr_mask

    def objective_simple(self, params: dict, n_objects: int, nn_number: int):
        """ Function for metric value evaluation """
        try:
            metrics = []
            for field_id in range(n_objects):
                if field_id % 100 == 0:
                    print(f'Process field number {field_id} (simple)')
                nn_forecasts, actual_matrix = self._get_predictions_from_networks(field_id)
                chip_forecasts = []
                weights = []
                for network_id in range(nn_number):
                    weight_label = f'weight_{network_id}'
                    threshold_label = f'th_{network_id}'

                    weight = params[weight_label]
                    threshold = params[threshold_label]

                    nn_mask = nn_forecasts[network_id]

                    # Binarization
                    nn_mask[nn_mask >= threshold] = 1
                    nn_mask[nn_mask < threshold] = 0
                    chip_forecasts.append(nn_mask)
                    weights.append(weight)

                stacked_prediction = np.stack(chip_forecasts)
                pr_mask = np.average(stacked_prediction, axis=0, weights=weights)

                pr_mask[pr_mask >= 0.5] = 1
                pr_mask[pr_mask < 0.5] = 0

                pr_mask = pr_mask.astype(np.uint8)

                # Prepare tensors
                pr_mask = torch.from_numpy(pr_mask)
                actual_mask = torch.from_numpy(actual_matrix)

                # Calculated metric
                iou_metric = smp.utils.metrics.IoU()
                calculated_metric = iou_metric.forward(pr_mask, actual_mask)
                metrics.append(-float(calculated_metric))

            metrics = np.array(metrics)
            return np.mean(metrics)
        except Exception as ex:
            return 10000.0

    def objective_extended(self, params: dict, n_objects: int, nn_number: int):
        """ Function for metric value evaluation in complicated search space """
        try:
            metrics = []
            for field_id in range(n_objects):
                if field_id % 100 == 0:
                    print(f'Process field number {field_id} (extended)')
                nn_forecasts, actual_matrix, chip_name = self._get_predictions_from_networks(field_id,
                                                                                             True)
                chip_metadata = self.metadata[self.metadata['chip_id'] == chip_name]
                # Datetime column has been transformed into month column
                current_time_location = str(chip_metadata['datetime'].values[0])

                chip_forecasts = []
                weights = []
                for network_id in range(nn_number):
                    weight_label = f'{current_time_location}_weight_{network_id}'
                    threshold_label = f'{current_time_location}_th_{network_id}'

                    weight = params[weight_label]
                    threshold = params[threshold_label]

                    nn_mask = nn_forecasts[network_id]

                    # Binarization
                    nn_mask[nn_mask >= threshold] = 1
                    nn_mask[nn_mask < threshold] = 0
                    chip_forecasts.append(nn_mask)
                    weights.append(weight)

                stacked_prediction = np.stack(chip_forecasts)
                pr_mask = np.average(stacked_prediction, axis=0, weights=weights)

                pr_mask[pr_mask >= 0.5] = 1
                pr_mask[pr_mask < 0.5] = 0

                pr_mask = pr_mask.astype(np.uint8)

                # Prepare tensors
                pr_mask = torch.from_numpy(pr_mask)
                actual_mask = torch.from_numpy(actual_matrix)

                # Calculated metric
                iou_metric = smp.utils.metrics.IoU()
                calculated_metric = iou_metric.forward(pr_mask, actual_mask)
                metrics.append(-float(calculated_metric))

            metrics = np.array(metrics)
            return np.mean(metrics)
        except Exception as ex:
            print(f'Metric valuation error: {ex.__str__()}')
            return 10000.0

    def _init_parameters_and_search_space(self):
        """ Initialize search space for optimization """
        if self.metadata is None:
            # Simple case
            network_files = [file for file in os.listdir(self.path) if '.pth' in file]
            equal_weight = 1.0 / len(network_files)
            for network_id, network_file in enumerate(network_files):
                weight_label = f'weight_{network_id}'
                threshold_label = f'th_{network_id}'
                self.parameters.update({weight_label: equal_weight,
                                        threshold_label: 0.5})

                # Initialize search space for all models
                self.search_space.update({weight_label: hp.uniform(weight_label, 0.05, 1.0),
                                          threshold_label: hp.uniform(threshold_label, 0.0001, 0.6)})
        else:
            self._init_extended_parameters_and_search_space()

    def _init_extended_parameters_and_search_space(self):
        """ Initialize network weights and network thresholds for every area separately """
        network_files = [file for file in os.listdir(self.path) if '.pth' in file]
        equal_weight = 1.0 / len(network_files)

        self.metadata['datetime'] = self.metadata['datetime'].dt.month
        months = self.metadata['datetime'].unique()
        print(f'Number of time locations: {len(months)}')
        for time_location in months:
            time_location = str(time_location)
            # For each time_location
            for network_id, network_file in enumerate(network_files):
                # For each neural network initialise it's own weight
                weight_label = f'{time_location}_weight_{network_id}'
                threshold_label = f'{time_location}_th_{network_id}'
                self.parameters.update({weight_label: equal_weight,
                                        threshold_label: 0.5})

                self.search_space.update({weight_label: hp.uniform(weight_label, 0.05, 1.0),
                                          threshold_label: hp.uniform(threshold_label, 0.0001, 0.6)})

    def __optimize_parameters(self, n_objects: int, iterations: int):
        """ Optimization step for current batch """
        nn_number = len(self.nn_models.keys())
        if self.metadata is None:
            best = fmin(partial(self.objective_simple,
                                n_objects=n_objects,
                                nn_number=nn_number),
                        space=self.search_space,
                        algo=tpe.suggest,
                        max_evals=iterations)
        else:
            best = fmin(partial(self.objective_extended,
                                n_objects=n_objects,
                                nn_number=nn_number),
                        space=self.search_space,
                        algo=tpe.suggest,
                        max_evals=iterations)

        # Get parameters
        best = space_eval(space=self.search_space, hp_assignment=best)
        self.parameters = best
