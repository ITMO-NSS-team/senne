import json
import os
import numpy as np

from senne.data.data import SenneDataLoader, train_test
from senne.data.preprocessing import normalize
from senne.log import senne_logger
from senne.repository.presets import *

TRAIN_SIZE = 0.85


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
        self.preprocessing_info = {}
        self.nn_models = {}

    def initialise_nn_models(self, preset: str):
        """ Create several neural networks for experiments """
        generate_function = self.networks_by_preset[preset]
        # Get list with parameters for neural network
        nn_params = generate_function()

        for i, data_info in self.preprocessing_info.items():
            current_network_parameters = nn_params[i]
            current_network_parameters['in_channels'] = data_info['in_channels']
            train_epoch, valid_epoch, nn_model = segmentation_net_builder(current_network_parameters)

            # Store models into the folder
            self.nn_models.update({i: {'train': [train_epoch, valid_epoch, nn_model],
                                       'batch_size': current_network_parameters['batch_size'],
                                       'epochs': current_network_parameters['epochs']}})

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

        # For every neural networks data has been already prepared - initialise networks
        self.initialise_nn_models(preset)

        # Fit neural networks and serialize models
        self._train_and_save()

    def apply_preprocessing(self, features_tensor: np.array, target_tensor: np.array,
                            preprocessing_names: list):
        """ Apply preprocessing on PyTorch tensors

        :param features_tensor: PyTorch tensor with source features matrices
        :param target_tensor: PyTorch tensor with source target matrices
        :param preprocessing_names: list with names of preprocessing strategies
        """
        preprocessing_information = {}
        for i, preprocessing_name in enumerate(preprocessing_names):
            copied_features = np.copy(features_tensor)
            copied_target = np.copy(target_tensor)

            # TODO implement different preprocessing strategies
            prep_folder = os.path.join(self.path, ''.join(('preprocessing_', str(i))))
            self._create_folder(prep_folder)

            # Apply normalization - default preprocessing for every strategy
            copied_features, copied_target, transformation_info = normalize(copied_features,
                                                                            copied_target)

            # Update shapes
            _, in_channels, _, _ = copied_features.shape
            self.preprocessing_info.update({i: {'path': prep_folder,
                                                'in_channels': in_channels}})

            # Save tensors in pt files
            copied_features = torch.from_numpy(copied_features)
            copied_target = torch.from_numpy(copied_target)
            torch.save(copied_features, os.path.join(prep_folder, 'features.pt'))
            torch.save(copied_target, os.path.join(prep_folder, 'target.pt'))

            dict_for_update = {'preprocessing_name': preprocessing_name,
                               'info': transformation_info}
            preprocessing_information.update({''.join(('network_', str(i))): dict_for_update})

        # Save information about preprocessing transformations
        json_path = os.path.join(self.path, 'preprocessing.json')
        with open(json_path, 'w') as f:
            json.dump(preprocessing_information, f)

    def _train_and_save(self):
        """ Train already initialized neural networks on prepared data """
        for i, data_info in self.preprocessing_info.items():
            # For every network
            neural_network_objects = self.nn_models[i]
            train_epoch, valid_epoch, nn_model = neural_network_objects['train']
            batch_size = neural_network_objects['batch_size']
            epochs = neural_network_objects['epochs']

            # Load data by path for current model
            data_path = self.preprocessing_info[i]['path']
            features_path = os.path.join(data_path, 'features.pt')
            target_path = os.path.join(data_path, 'target.pt')

            features = torch.load(features_path)
            features = features.float()

            target = torch.load(target_path)

            # Split tensors and create datasets
            train_dataset, valid_dataset = train_test(features, target,
                                                      train_size=TRAIN_SIZE)

            # Prepare data loaders
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size)
            valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)

            # Launch training
            path_to_save = os.path.join(self.path, f'network_{i}.pth')
            checkpoints = [20, 30, 40, 50, 60, 70, 80, 90, 100, 150]
            for epoch_number in range(0, epochs):

                print('\nEpoch: {}'.format(epoch_number))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)

                if any(epoch_number == checkpoint for checkpoint in checkpoints):
                    senne_logger.info(f'Model {i} was saved for epoch {epoch_number}!')
                    torch.save(nn_model, path_to_save)

            torch.save(nn_model, path_to_save)
            senne_logger.info(f'Model {i} was saved!')

    def prepare_composite_model(self, final_model: str):
        """ Start creating composite ensemble with several neural networks

        :param final_model: which model to use for ensembling.
        Available parameters:
            * ridge - ridge model for ensembling
            * lasso - lasso regression
        """

    @staticmethod
    def _create_folder(path):
        if os.path.isdir(path) is False:
            os.makedirs(path)
