import json
import os

import pandas as pd
import numpy as np
import torch.utils.data as data_utils
from fedot.api.main import Fedot
from sklearn.metrics import classification_report

from senne.data.data import SenneDataLoader, train_test_torch, train_test_numpy
from senne.data.preprocessing import normalize, apply_normalization
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
    Possible combination is 'filter_max | remote_cloud | augmentation' etc.
    """
    networks_by_preset = {'two_simple': create_two_simple_networks,
                          'three_simple': create_three_simple_networks,
                          'four_simple': create_four_simple_networks,
                          'two_advanced': create_two_advanced_networks}

    preprocessing_by_preset = {'two_simple': ['default', 'default'],
                               'three_simple': ['default', 'augmentation',
                                                'filter_max'],
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

        # Sampling parameters
        self.sampling_ratio = None
        self.sampling_ids = None
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

        :param data_paths: dictionary with paths to features matrices and target ones
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

    def prepare_composite_model(self, data_paths: dict, final_model: str, sampling_ratio: float = 0.01):
        """ Start creating composite ensemble with several neural networks

        :param data_paths: dictionary with paths to features matrices and target ones
        :param final_model: which model to use for ensembling.
        Available parameters:
            * logit - logistic regression (in progress)
            * dt - decision tree classification (in progress)
            * automl - launch FEDOT framework as core
        :param sampling_ratio: which ratio of training sample need to use for training
        """
        self.sampling_ratio = sampling_ratio
        self.data_loader = SenneDataLoader(features_path=data_paths['features_path'],
                                           target_path=data_paths['target_path'])
        train_df, test_df = self.collect_predictions_from_networks()
        features_column = train_df.columns[:-1]
        if final_model == 'automl':
            # Launch FEDOT framework
            senne_logger.info('Launch AutoML algorithm')

            # task selection, initialisation of the framework
            automl_model = Fedot(problem='classification', timeout=10)

            # Define parameters and start optimization

            pipeline = automl_model.fit(features=np.array(train_df[features_column]),
                                        target=np.array(train_df['target']),
                                        predefined_model='logit')

            #################
            # Save pipeline #
            #################
            pipeline.save(path=os.path.join(self.path, 'final_model'))
            folders = os.listdir(self.path)
            for folder in folders:
                if folder.endswith('final_model'):
                    # Folder need to be renamed
                    old_name = os.path.join(self.path, folder)
                    new_name = os.path.join(self.path, 'final_model')
                    os.rename(old_name, new_name)

            # Display validation metrics
            predictions = automl_model.predict(np.array(test_df[features_column]))
            print(classification_report(test_df['target'], predictions))

    def collect_predictions_from_networks(self):
        """ Apply already fitted preprocessing """
        preprocess_json = os.path.join(self.path, 'preprocessing.json')
        with open(preprocess_json) as json_file:
            preprocess_info = json.load(json_file)

        number_of_models = len(preprocess_info.keys())
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for model_id in range(number_of_models):
            network_name = ''.join(('network_', str(model_id)))
            current_model_info = preprocess_info[network_name]

            # Prepare data for neural networks
            features_tensor, target_tensor = self.data_loader.get_numpy_arrays()
            train_dataset, test_dataset = prepare_data_for_model(features_tensor,
                                                                 target_tensor,
                                                                 current_model_info)

            # Make predictions
            model_path = os.path.join(self.path, ''.join((network_name, '.pth')))
            train_predictions, test_predictions = get_predictions_from_networks(train_dataset, test_dataset,
                                                                                model_path, self.device)

            train_df[network_name] = self._return_sample_for_train(train_predictions)
            test_df[network_name] = self._return_sample_for_test(test_predictions)

            if model_id == number_of_models - 1:
                # Use row and column id as additional predictions
                train_df['row_ids'], train_df['col_ids'] = create_row_column_features(train_predictions,
                                                                                      sampling_ids=self.sampling_ids)
                test_df['row_ids'], test_df['col_ids'] = create_row_column_features(test_predictions)

                # Add target columns
                train_target = np.ravel(train_dataset.tensors[1].cpu().numpy())
                train_df['target'] = train_target[self.sampling_ids]

                test_df['target'] = np.ravel(test_dataset.tensors[1].cpu().numpy())

        senne_logger.info(f'Training sample size {len(train_df)}')
        return train_df, test_df

    @staticmethod
    def _create_folder(path):
        if os.path.isdir(path) is False:
            os.makedirs(path)

    def _return_sample_for_train(self, train_predictions: np.array):
        """ Convert stack with matrices into one dimensional array """
        one_dim_array = np.ravel(train_predictions)

        if self.sampling_ids is None:
            n_pixels = len(one_dim_array)
            n_pixels_to_take = round(n_pixels * self.sampling_ratio)
            # Define sampling ids based on ratio
            source_ids = np.arange(0, n_pixels)
            np.random.shuffle(source_ids)
            self.sampling_ids = source_ids[: n_pixels_to_take]

        one_dim_array = np.ravel(train_predictions)
        return one_dim_array[self.sampling_ids]

    def _return_sample_for_test(self, test_predictions: np.array):
        return np.ravel(test_predictions)

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
            train_dataset, valid_dataset = train_test_torch(features, target,
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


def prepare_data_for_model(features_tensor: np.array, target_tensor: np.array,
                           preprocess_info: dict):
    """ Prepare pytorch tensors in a form of datasets """
    # TODO use preprocessors classes
    train_fs, test_fs, train_target, test_target = train_test_numpy(features_tensor,
                                                                    target_tensor)

    # Normalize data
    train_fs = apply_normalization(train_fs, preprocess_info)
    test_fs = apply_normalization(test_fs, preprocess_info)

    # Convert arrays into PyTorch tensors
    train_fs = torch.from_numpy(train_fs)
    test_fs = torch.from_numpy(test_fs)
    train_target = torch.from_numpy(train_target)
    test_target = torch.from_numpy(test_target)

    # Create Datasets for train and validation
    train_dataset = data_utils.TensorDataset(train_fs, train_target)
    test_dataset = data_utils.TensorDataset(test_fs, test_target)
    return train_dataset, test_dataset


def get_predictions_from_networks(train_dataset: data_utils.TensorDataset,
                                  test_dataset: data_utils.TensorDataset,
                                  model_path: str, device: str):
    """
    Make a prediction by corresponding neural network
    """
    # Load PyTorch neural network
    nn_model = torch.load(model_path)

    predicted_train_masks = _predict_on_dataset(nn_model, device, train_dataset)
    predicted_test_masks = _predict_on_dataset(nn_model, device, test_dataset)

    return predicted_train_masks, predicted_test_masks


def _predict_on_dataset(nn_model, device: str, dataset: data_utils.TensorDataset):
    """ Iterative prediction from neural network """
    features_tensor = dataset.tensors[0]
    n_objects, _, _, _ = features_tensor.size()
    predicted_masks = []
    for i in range(n_objects):
        current_features = features_tensor[i, :, :, :]
        pr_mask = nn_model.predict(current_features.to(device).unsqueeze(0))
        # Into numpy array
        pr_mask = pr_mask.squeeze().cpu().numpy()

        predicted_masks.append(pr_mask)

    return np.array(predicted_masks)


def create_row_column_features(predictions: np.array, sampling_ids: np.array = None):
    """ Create two columns (features) with indices

    :param predictions: output from neural network in a form of multidimensional array
    :param sampling_ids: indices to take
    """
    n_objects, n_rows, n_cols = predictions.shape
    # Column indices
    row_ids = np.arange(0, n_rows).reshape((-1, 1))
    row_ids = np.repeat(row_ids, n_cols, axis=1)
    all_row_matrix = np.ravel(np.array([row_ids] * n_objects))

    # Column indices
    col_ids = np.arange(0, n_cols).reshape((1, -1))
    col_ids = np.repeat(col_ids, n_rows, axis=0)
    all_col_matrix = np.ravel(np.array([col_ids] * n_objects))

    if sampling_ids is None:
        return [all_row_matrix, all_col_matrix]
    else:
        sampled_row_ids = all_row_matrix[sampling_ids]
        sampled_col_ids = all_col_matrix[sampling_ids]
        return [sampled_row_ids, sampled_col_ids]
