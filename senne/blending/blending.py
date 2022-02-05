import os
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import torch

from senne.data.data import SenneDataset
from senne.data.preprocessing import apply_normalization


class AbstractEnsemble(ABC):
    """ Base class for collecting predictions from several neural networks """

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

        self.for_predict = for_predict
        if self.for_predict:
            # Ensemble was initialized to make predictions - load neural networks
            self._init_networks_models()

    @abstractmethod
    def fit(self, **kwargs):
        """ Perform ensemble model training """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, chip_arr: np.array, chip_name: str = None, use_network: int = None):
        """ Perform predict for ensemble with several neural networks """
        raise NotImplementedError()

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

    def _preprocess_field(self, network_file: str, features_tensor: np.array) -> np.array:
        """ Perform preprocessing """
        # TODO extend preprocessing
        preprocessing_description = self.networks_info[network_file]
        # Default preprocessing for all neural networks - normalization
        features_tensor = apply_normalization(features_tensor, self.boundaries_info)
        return features_tensor
