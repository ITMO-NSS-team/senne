import json
import os
import pickle

import pandas as pd
import numpy as np
import torch.utils.data as data_utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from senne.data.data import DataProcessor, TRAIN_SIZE, SenneDataset
from senne.data.preprocessing import normalize, apply_normalization
from senne.log import senne_logger
from senne.repository.presets import *


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
        self.sampling_ratio = sampling_ratio
        self.data_processor = DataProcessor(features_path=data_paths['features_path'],
                                            target_path=data_paths['target_path'])

        if final_model == 'weighted':
            raise NotImplementedError(f'Weighted model in progress')
        else:
            train_df, test_df = self.collect_predictions_from_networks()
            features_column = train_df.columns[:-1]
            if final_model == 'automl':
                from fedot.api.main import Fedot

                # Launch FEDOT framework
                senne_logger.info('Launch AutoML algorithm')

                # task selection, initialisation of the framework
                automl_model = Fedot(problem='classification', timeout=10)

                # Define parameters and start optimization
                pipeline = automl_model.fit(features=np.array(train_df[features_column]),
                                            target=np.array(train_df['target']))
                predictions = automl_model.predict(np.array(test_df[features_column]))

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

            elif final_model == 'logit':
                logit = LogisticRegression()
                logit.fit(np.array(train_df[features_column]), np.array(train_df['target']))
                predictions = logit.predict(np.array(test_df[features_column]))

                save_path = os.path.join(self.path, 'final_model.pkl')
                with open(save_path, "wb") as f:
                    pickle.dump(logit, f)
            else:
                raise NotImplementedError(f'Model {final_model} can not be used')

            # Display validation metrics
            print(classification_report(test_df['target'], predictions))

    def collect_predictions_from_networks(self):
        """ Apply already fitted preprocessing """
        boundaries_info, networks_info = self._load_json_files()

        # Load dataframes with paths
        train_paths = pd.read_csv(os.path.join(self.path, 'train.csv'))
        test_paths = pd.read_csv(os.path.join(self.path, 'test.csv'))

        network_files = [file for file in os.listdir(self.path) if '.pth' in file]
        number_of_models = len(network_files)
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for model_id, network_name in enumerate(network_files):
            # Make predictions
            model_path = os.path.join(self.path, network_name)
            current_preprocessing = networks_info[network_name]

            train_dataset = SenneDataset(serialized_folder=self.path,
                                         dataframe_with_paths=train_paths,
                                         transforms=current_preprocessing)
            # For validation there is no need to perform some calculations
            test_dataset = SenneDataset(serialized_folder=self.path,
                                        dataframe_with_paths=test_paths,
                                        transforms=current_preprocessing,
                                        for_train=False)
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

    def _load_json_files(self):
        boundaries_json = os.path.join(self.path, 'boundaries.json')
        with open(boundaries_json) as json_file:
            boundaries_info = json.load(json_file)

        networks_json = os.path.join(self.path, 'networks_info.json')
        with open(networks_json) as json_file:
            networks_info = json.load(json_file)

        return boundaries_info, networks_info

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

                print('\nEpoch: {}'.format(epoch_number))
                train_logs = train_epoch.run(train_loader)
                valid_logs = valid_epoch.run(valid_loader)

                if any(epoch_number == checkpoint for checkpoint in checkpoints):
                    senne_logger.info(f'Model {i} was saved for epoch {epoch_number}!')
                    torch.save(nn_model, path_to_save)

            torch.save(nn_model, path_to_save)
            senne_logger.info(f'Model {i} was saved!')

        json_path = os.path.join(self.path, 'networks_info.json')
        with open(json_path, 'w') as f:
            json.dump(network_train_info, f)


def get_predictions_from_networks(train_dataset: SenneDataset,
                                  test_dataset: SenneDataset,
                                  model_path: str, device: str):
    """
    Make a prediction by corresponding neural network
    """
    # Load PyTorch neural network
    nn_model = torch.load(model_path)

    predicted_train_masks = _predict_on_dataset(nn_model, device, train_dataset)
    predicted_test_masks = _predict_on_dataset(nn_model, device, test_dataset)

    return predicted_train_masks, predicted_test_masks


def _predict_on_dataset(nn_model, device: str, dataset: SenneDataset):
    """ Iterative prediction from neural network """
    n_objects = len(dataset)
    predicted_masks = []
    for i in range(n_objects):
        current_features, current_target = dataset.__getitem__(index=i)
        pr_mask = nn_model.predict(current_features.to(device).unsqueeze(0))
        # Into numpy array
        pr_mask = pr_mask.squeeze().cpu().numpy()

        predicted_masks.append(pr_mask)

        import matplotlib.pyplot as plt
        pr_mask[pr_mask < 0.2] = 0
        pr_mask[pr_mask >= 0.2] = 0
        plt.imshow(pr_mask)
        plt.colorbar()
        plt.show()

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
