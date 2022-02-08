import os
import pickle
import timeit

import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from senne.blending.blending import AbstractEnsemble
from senne.log import senne_logger


class MLEnsemble(AbstractEnsemble):
    """ Class for ensembling neural networks prediction based on machine learning models """

    model_by_name = {'logit': LogisticRegression,
                     'rf': RandomForestClassifier,
                     'svc': SVC,
                     'naive_bayes': GaussianNB}

    def __init__(self, model_name: str, boundaries_info: dict, networks_info: dict, path: str,
                 device: str, metadata_path: str = None, for_predict: bool = False,
                 use_coordinates: bool = False, use_shift: bool = False):
        super().__init__(boundaries_info, networks_info, path, device, metadata_path, for_predict)
        self.model_name = model_name
        self.use_shift = use_shift

        # Is there a need to use coordinates as features to train ensemble
        self.use_coordinates = use_coordinates
        if for_predict:
            # Load serialized model
            load_path = os.path.join(self.path, 'final_model.pkl')
            with open(load_path, "rb") as file:
                self.ensemble_model = pickle.load(file)

    def fit(self, sampling_ratio: float):
        """ Perform machine learning model training """
        df_paths = pd.read_csv(os.path.join(self.path, 'test.csv'))
        n_objects = len(df_paths)

        self._init_networks_dataset(df_paths)
        self._init_networks_models()

        for field_id in range(n_objects):
            features_matrix = self._get_source_features(field_id)
            nn_forecasts, actual_matrix = self._get_predictions_from_networks(field_id)

            if field_id == 0:
                # Initialize dataframe with features
                train_dataframe = self._prepare_table_dataframe(features_matrix, nn_forecasts,
                                                                actual_matrix, sampling_ratio)
            else:
                new_dataframe = self._prepare_table_dataframe(features_matrix, nn_forecasts,
                                                                actual_matrix, sampling_ratio)
                train_dataframe = pd.concat([train_dataframe, new_dataframe])

        if self.use_shift:
            train_dataframe = create_shifted_features(train_dataframe)

        if self.use_shift is True and sampling_ratio is not None:
            # There is a need to prepare "sparse" dataframe due to it wasn't prepared previously
            train_dataframe = sample_train_data(train_dataframe, sampling_ratio)

        # Train appropriate algorithm
        senne_logger.info(f'Start to train ensemble model on {len(train_dataframe)} objects')
        # Initialize model
        class_model = self.model_by_name[self.model_name]()

        # Take table without last column (target)
        train_array = train_dataframe.values[:, :-1]
        train_target = np.array(train_dataframe['target'])
        class_model.fit(train_array, train_target)

        # Serialize model
        train_predicted = class_model.predict(train_array)
        senne_logger.info(f'F1: {f1_score(train_target, train_predicted):4f}')

        save_path = os.path.join(self.path, 'final_model.pkl')
        with open(save_path, "wb") as f:
            pickle.dump(class_model, f)

    def predict(self, chip_arr: np.array, chip_name: str = None, use_network: int = None):
        """ Perform predict based on ensemble model """
        # Take month from datetime and store in into datetime column
        if chip_name is not None:
            self.metadata['datetime'] = self.metadata['datetime'].dt.month
        network_files = [file for file in os.listdir(self.path) if '.pth' in file]
        network_files.sort()

        # Collect predictions from neural networks
        predicted_probabilities_field = []
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
            predicted_probabilities_field.append(pr_mask)

        predicted_probabilities_field = np.array(predicted_probabilities_field, dtype=float)

        # Take features matrices
        features_matrix = self._preprocess_field(network_files[0], chip_arr)
        n_objects, n_rows, n_cols = features_matrix.shape
        features_for_predict = self._prepare_table_dataframe(features_matrix=features_matrix,
                                                             nn_forecasts=predicted_probabilities_field,
                                                             actual_matrix=None, sampling_ratio=None)
        if self.use_shift:
            features_for_predict = create_shifted_features(features_for_predict, for_predict=True)

        start = timeit.default_timer()
        predicted = self.ensemble_model.predict(features_for_predict.values)
        predicted = predicted.reshape(n_rows, n_cols)
        predicted = predicted.astype(np.uint8)
        print('Predict time', timeit.default_timer() - start)

        return predicted

    def _prepare_table_dataframe(self, features_matrix: np.array, nn_forecasts: np.array,
                                 actual_matrix: np.array, sampling_ratio: float = None) -> pd.DataFrame:
        """ Create dataframe in tabular form for time series

        :param features_matrix:
        :param nn_forecasts:
        :param actual_matrix:
        :param sampling_ratio:
        """
        tabular_data = {}
        # Collect features from source matrices
        for feature_matrix_id in range(len(features_matrix)):
            current_features_field = features_matrix[feature_matrix_id]
            tabular_data.update({f'feature_{feature_matrix_id}': np.ravel(current_features_field)})

        # Collect predictions from neural networks
        for forecast_matrix_id in range(len(nn_forecasts)):
            current_forecast_field = nn_forecasts[forecast_matrix_id]
            tabular_data.update({f'forecast_{forecast_matrix_id}': np.ravel(current_forecast_field)})

        # Prepare target column if necessary
        if actual_matrix is not None:
            tabular_data.update({'target': np.ravel(actual_matrix)})

        tabular_data = pd.DataFrame(tabular_data)
        # Perform sampling only for train and if no shifting is planned
        if sampling_ratio is not None and actual_matrix is not None and self.use_shift is False:
            tabular_data = sample_train_data(tabular_data, sampling_ratio)

        return tabular_data

    def _get_source_features(self, field_id: int):
        """ Return features matrices as for first neural network """
        dataset = self.nn_datasets[0]
        current_features, _ = dataset.__getitem__(index=field_id)
        current_features = current_features.squeeze().cpu().numpy()
        return current_features


def sample_train_data(tabular_data, sampling_ratio):
    """ Take small part of pixels from dataframe """
    n_pixels = len(tabular_data)
    n_pixels_to_take = round(n_pixels * sampling_ratio)

    # Define sampling ids to take based on ratio
    source_ids = np.arange(0, n_pixels)
    np.random.shuffle(source_ids)
    sampling_ids = source_ids[: n_pixels_to_take]
    # Take only a part of pixels
    tabular_data = tabular_data.iloc[sampling_ids]

    return tabular_data


def create_shifted_features(train_dataframe: pd.DataFrame, for_predict: bool = False):
    """ Take neighboring pixels values as additional features

    Pixels
    a1 | a2 | a3
    -------------
    a4 | a5 | a6
    -------------
    a7 | a8 | a9

    For a5 values from a4 and a6 will be used
    """
    start = timeit.default_timer()
    # TODO change
    matrix_side_size = 512

    if for_predict:
        features = train_dataframe
    else:
        # Take only features table
        features_columns = train_dataframe.columns[:-1]
        features = train_dataframe[features_columns]

    neighboring_left_side = features.shift(1)
    neighboring_right_side = features.shift(-1)

    neighboring_left_side.iloc[0, :] = features.iloc[0, :]
    neighboring_right_side.iloc[-1, :] = features.iloc[-1, :]

    # Correct border pixels for horizontal propagation
    almost_all_matrix_elements = (matrix_side_size * matrix_side_size) - 1
    ids = np.arange(matrix_side_size - 1, almost_all_matrix_elements, matrix_side_size)
    ids_plus = np.arange(matrix_side_size, almost_all_matrix_elements, matrix_side_size)
    neighboring_right_side.iloc[ids, :] = features.iloc[ids, :]
    neighboring_left_side.iloc[ids_plus, :] = features.iloc[ids_plus, :]

    # Take only source features and first neural network prediction
    train_dataframe = pd.concat([neighboring_left_side, neighboring_right_side, train_dataframe], axis=1)
    print(timeit.default_timer() - start)
    return train_dataframe
