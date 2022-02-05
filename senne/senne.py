import json
import os

import pandas as pd
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

from senne.blending.ml import MLEnsemble
from senne.data.data import DataProcessor, TRAIN_SIZE, SenneDataset
from senne.log import senne_logger
from senne.repository.presets import *
from senne.blending.weighted import WeightedEnsemble


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
        * filter_min - filter images and stay only images with low cloud ratio (less than 30%)
        * filter_max - filter images and stay only images with high cloud ratio (more than 70%)
        * filter_steady - filter images and stay only images with cloud ratio more than 20% and
        less than 80%
        * remote_cloud - use advanced method from remote sensing to calculate special indices

    Sign ' | ' applied for composite preprocessing pipelines with several stages.
    Possible combination is 'filter_max | remote_cloud | augmentation' etc.
    """
    networks_by_preset = {'two_simple': create_two_simple_networks,
                          'three_simple': create_three_simple_networks,
                          'four_simple': create_four_simple_networks,
                          'two_advanced': create_two_advanced_networks}

    preprocessing_by_preset = {'two_simple': ['default', 'filter_steady'],
                               'three_simple': ['default', 'augmentation',
                                                'filter_max'],
                               'four_simple': ['default', 'augmentation',
                                               'filter_min', 'filter_max'],
                               'two_advanced': ['filter_min | augmentation',
                                                'filter_max | remote_cloud | augmentation']}

    def __init__(self, path: str, metadata_path: str = None, device: str = 'cuda'):
        self.path = os.path.abspath(path)
        if metadata_path is not None:
            self.metadata_path = os.path.abspath(metadata_path)
        else:
            self.metadata_path = None
        # Create folder if necessary
        self._create_folder(self.path)

        self.device = device

        # Classes for data preprocessing
        self.sampling_ratio = None
        self.sampling_ids = None
        self.data_processor = None
        self.remote_preprocessor = None
        self.image_preprocessor = None
        self.preset = None

        self.nn_models = {}

    def train_neural_networks(self, data_paths: dict, preset: str):
        """ Train several neural networks for image segmentation

        :param data_paths: dictionary with paths to features matrices and target ones
        :param preset: define which neural networks to use for ensembling
        """
        # Load data and prepare train test sample
        self.data_processor = DataProcessor(features_path=data_paths['features_path'],
                                            target_path=data_paths['target_path'])
        self.data_processor.collect_sample_info(serialized_path=self.path)

        self.initialise_nn_models(preset)

        # Fit neural networks and serialize models
        self._train_and_save()

    def initialise_nn_models(self, preset: str):
        """ Create several neural networks for experiments """
        # TODO: refactor to create networks iteratively
        self.preset = preset
        generate_function = self.networks_by_preset[preset]
        # Get list with parameters for neural network
        n_networks_params = generate_function()

        for i, current_network_parameters in enumerate(n_networks_params):
            current_network_parameters['in_channels'] = self.data_processor.in_channels(self.path)
            train_epoch, valid_epoch, nn_model = segmentation_net_builder(current_network_parameters)

            # Store models into the folder
            self.nn_models.update({i: {'train': [train_epoch, valid_epoch, nn_model],
                                       'batch_size': current_network_parameters['batch_size'],
                                       'epochs': current_network_parameters['epochs']}})

    def prepare_composite_model(self, data_paths: dict, final_model: str, sampling_ratio: float = 0.01):
        """ Start creating composite ensemble with several neural networks

        :param data_paths: dictionary with paths to features matrices and target ones
        :param final_model: which model to use for ensembling.
        Available parameters:
            * logit - logistic regression
            * weighted - weighted average with thresholds
            * automl - launch FEDOT framework as a core
        :param sampling_ratio: which ratio of training sample need to use for training
        """
        self.data_processor = DataProcessor(features_path=data_paths['features_path'],
                                            target_path=data_paths['target_path'])

        boundaries_info, networks_info = load_json_files(self.path)
        if final_model == 'weighted':
            weighted_model = WeightedEnsemble(boundaries_info, networks_info,
                                              path=self.path, device=self.device,
                                              metadata_path=self.metadata_path)
            weighted_model.fit(50)
            # Save parameters
            json_path = os.path.join(self.path, 'ensemble_info.json')
            with open(json_path, 'w') as f:
                json.dump(weighted_model.parameters, f)
        else:
            ml_model = MLEnsemble(final_model, boundaries_info, networks_info,
                                  path=self.path, device=self.device,
                                  metadata_path=self.metadata_path)
            ml_model.fit(sampling_ratio)

    @staticmethod
    def _create_folder(path):
        if os.path.isdir(path) is False:
            os.makedirs(path)

    def _train_and_save(self):
        """ Train already initialized neural networks on prepared data """
        # Get information about preprocessing
        preprocessing_to_apply = self.preprocessing_by_preset[self.preset]

        network_train_info = {'preset': self.preset}
        for i, neural_network_objects in self.nn_models.items():
            current_preprocessing = preprocessing_to_apply[i]

            # For every network
            train_epoch, valid_epoch, nn_model = neural_network_objects['train']
            batch_size = neural_network_objects['batch_size']
            epochs = neural_network_objects['epochs']

            # Load pandas Dataframe with paths to files
            full_df = pd.read_csv(os.path.join(self.path, 'train.csv'))
            train_df, valid_df = train_test_split(full_df, train_size=TRAIN_SIZE)

            # Filter train data if it's required
            train_df = filter_data(train_df, current_preprocessing)

            # Create SENNE datasets
            train_dataset = SenneDataset(serialized_folder=self.path,
                                         dataframe_with_paths=train_df,
                                         transforms=current_preprocessing)
            # For validation there is no need to perform some calculations
            valid_dataset = SenneDataset(serialized_folder=self.path,
                                         dataframe_with_paths=valid_df,
                                         transforms=current_preprocessing,
                                         for_train=False)

            # Prepare data loaders
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size)
            valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)

            # Launch training
            path_to_save = os.path.join(self.path, f'network_{i}.pth')
            network_train_info.update({f'network_{i}.pth': current_preprocessing})
            checkpoints = [20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
            for epoch_number in range(0, epochs):
                try:
                    print('\nEpoch: {}'.format(epoch_number))
                    train_logs = train_epoch.run(train_loader)
                    valid_logs = valid_epoch.run(valid_loader)

                    if any(epoch_number == checkpoint for checkpoint in checkpoints):
                        senne_logger.info(f'Model {i} was saved for epoch {epoch_number}!')
                        torch.save(nn_model, path_to_save)

                except RuntimeError as ex:
                    senne_logger.info(f'Model {i} was saved for epoch {epoch_number}!')
                    torch.save(nn_model, path_to_save)
                    print(f'RuntimeError occurred {ex.__str__()}. Continue')
                    continue

            torch.save(nn_model, path_to_save)
            senne_logger.info(f'Model {i} was saved!')

        json_path = os.path.join(self.path, 'networks_info.json')
        with open(json_path, 'w') as f:
            json.dump(network_train_info, f)


def load_json_files(path: str):
    path = os.path.abspath(path)
    boundaries_json = os.path.join(path, 'boundaries.json')
    with open(boundaries_json) as json_file:
        boundaries_info = json.load(json_file)

    networks_json = os.path.join(path, 'networks_info.json')
    with open(networks_json) as json_file:
        networks_info = json.load(json_file)

    return boundaries_info, networks_info


def filter_data(train_df: pd.DataFrame, current_preprocessing: str):
    """ Filter training sample by cloud conditions """
    if 'filter_min' in current_preprocessing:
        # Remain only images with low cloud ratio (less than 30%)
        train_df = train_df[train_df['cloud_ratio'] <= 0.3]
    elif 'filter_steady' in current_preprocessing:
        # Cloud ratio between 20 and 80%
        train_df = train_df[train_df['cloud_ratio'] >= 0.2]
        train_df = train_df[train_df['cloud_ratio'] <= 0.8]
    elif 'filter_max' in current_preprocessing:
        # Remain only images with high cloud ratio (more than 70%)
        train_df = train_df[train_df['cloud_ratio'] >= 0.7]

    return train_df
