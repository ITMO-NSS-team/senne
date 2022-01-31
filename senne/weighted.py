import os
import pickle

import pandas as pd
import numpy as np
import torch
import segmentation_models_pytorch as smp
from functools import partial

from hyperopt import hp, fmin, tpe, space_eval
from senne.data.data import SenneDataset
from senne.data.preprocessing import apply_normalization
from senne.log import senne_logger


class WeightedEnsemble:
    """ Class for ensemble forecasts from several neural networks """

    def __init__(self, boundaries_info: dict, networks_info: dict, path: str,
                 device: str, metadata_path: str = None, for_predict: bool = False):
        self.boundaries_info = boundaries_info
        self.networks_info = networks_info
        self.path = os.path.abspath(path)
        if metadata_path is not None:
            # Read metadata dataframe
            self.metadata = pd.read_csv(os.path.abspath(metadata_path), parse_dates=['datetime'])
        else:
            self.metadata = None
        self.device = device

        self.nn_datasets = {}
        self.nn_models = {}
        self.parameters = {}
        self.search_space = {}

        self.for_predict = for_predict
        if self.for_predict:
            # Ensemble was initialized to make predictions - load neural networks
            self._init_networks_models()

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

    def predict(self, chip_arr: np.array, chip_name: str = None):
        """ Perform predict for ensemble with several neural networks """
        # Take month from datetime and store in into datetime column
        if chip_name is not None:
            self.metadata['datetime'] = self.metadata['datetime'].dt.month
        network_files = [file for file in os.listdir(self.path) if '.pth' in file]

        predicted_masks = []
        weights = []
        for network_id, network_file in enumerate(network_files):
            features_tensor = np.copy(chip_arr)
            # Perform preprocessing for chip
            features_tensor = self._preprocess_field(network_file, features_tensor)

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
                chip_metadata = self.metadata[self.metadata['chip_id'] == chip_name]
                current_time_location = str(chip_metadata['datetime'].values[0])

                threshold = self.parameters[f'{current_time_location}_th_{network_id}']
                pr_mask[pr_mask >= threshold] = 1
                pr_mask[pr_mask < threshold] = 0

                predicted_masks.append(pr_mask)
                weights.append(self.parameters[f'{current_time_location}_weight_{network_id}'])

        predicted_masks = np.array(predicted_masks, dtype=float)
        pr_mask = np.average(predicted_masks, axis=0, weights=weights)

        pr_mask[pr_mask >= 0.5] = 1
        pr_mask[pr_mask < 0.5] = 0

        pr_mask = pr_mask.astype(np.uint8)

        return pr_mask

    def _init_networks_dataset(self, df_paths: pd.DataFrame):
        """ Initialize datasets for each neural network """
        network_files = [file for file in os.listdir(self.path) if '.pth' in file]
        for network_id, network_file in enumerate(network_files):
            current_preprocessing = self.networks_info[network_file]
            dataset = SenneDataset(serialized_folder=self.path,
                                   dataframe_with_paths=df_paths,
                                   transforms=current_preprocessing,
                                   for_train=False)
            self.nn_datasets.update({network_id: dataset})

    def _init_networks_models(self):
        """ Load all neural networks for ensembling """
        network_files = [file for file in os.listdir(self.path) if '.pth' in file]
        for network_id, network_file in enumerate(network_files):
            network_path = os.path.join(self.path, network_file)
            nn_model = torch.load(network_path)
            nn_model = nn_model.to(self.device)

            self.nn_models.update({network_id: nn_model})

    def _get_predictions_from_networks(self, field_id: int,
                                       is_chip_name_needed: bool = False):
        """ Launch neural networks to make forecasts """
        dataset = None
        networks_ids = list(self.nn_datasets.keys())
        networks_ids.sort()

        predicted_masks = []
        current_target = None
        for network_id in networks_ids:
            # Take pre-loaded neural network
            nn_model = self.nn_models[network_id]
            # And use corresponding dataset
            dataset = self.nn_datasets[network_id]

            current_features, current_target = dataset.__getitem__(index=field_id)
            pr_mask = nn_model.predict(current_features.to(self.device).unsqueeze(0))
            # Into numpy array
            pr_mask = pr_mask.squeeze().cpu().numpy()
            predicted_masks.append(pr_mask)

        predicted_masks = np.array(predicted_masks)
        if is_chip_name_needed:
            # Take last dataset and find name if chip
            chip_name = dataset.get_chip_name_by_id(index=field_id)
            return predicted_masks, current_target.squeeze().cpu().numpy(), chip_name
        else:
            return predicted_masks, current_target.squeeze().cpu().numpy()

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

    def _preprocess_field(self, network_file: str, features_tensor: np.array) -> np.array:
        """ Perform preprocessing """
        # TODO extend preprocessing
        preprocessing_description = self.networks_info[network_file]
        # Default preprocessing for all neural networks - normalization
        features_tensor = apply_normalization(features_tensor, self.boundaries_info)
        return features_tensor

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
