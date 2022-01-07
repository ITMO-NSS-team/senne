import json
import pickle
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
import cv2 as cv
import os

import torch
from tifffile import imsave, imread

ROOT_DIRECTORY = Path("/codeexecution")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
INPUT_IMAGES_DIRECTORY = ROOT_DIRECTORY / "data/test_features"
SERIALIZED_MODELS_DIR = ROOT_DIRECTORY / "serialized"

BANDS = ["B02", "B03", "B04", "B08"]
DEVICE = 'cuda'
chip_ids = (pth.name for pth in INPUT_IMAGES_DIRECTORY.iterdir() if not pth.name.startswith("."))


def make_composite_prediction(features_tensor, nn_models: dict, final_model):
    """ The main idea is to use ensemble model to make accurate predictions """
    serialized_path = str(SERIALIZED_MODELS_DIR)
    files = os.listdir(str(SERIALIZED_MODELS_DIR))

    # Load json with preprocessing info
    preprocess_json = os.path.join(serialized_path, 'preprocessing.json')
    with open(preprocess_json) as json_file:
        preprocess_info = json.load(json_file)

    network_names = list(nn_models.keys())
    network_names.sort()

    # Take predictions from each neural network
    features_df = {}
    for i, network_name in enumerate(network_names):
        features_copied = np.copy(features_tensor)

        current_model_info = preprocess_info[network_name.split('.')[0]]
        features_copied = _prepare_data_for_predict(features_copied,
                                                    current_model_info)
        current_model = nn_models[network_name]

        # Get neural networks output
        pr_mask = current_model.predict(features_copied.to(DEVICE))
        pr_mask = pr_mask.squeeze().cpu().numpy()

        # Update information about predictions
        features_df[network_name] = np.ravel(pr_mask)

        if i == len(network_names) - 1:
            # Last neural network give prediction
            all_rows, all_cols = _create_row_column_features_for_predict(pr_mask)
            features_df['row_ids'] = all_rows
            features_df['col_ids'] = all_cols

    features_df = pd.DataFrame(features_df)
    features_names = copy(network_names)
    features_names.extend(['row_ids', 'col_ids'])

    labels_output = final_model.predict(np.array(features_df[features_names]))
    labels_output = labels_output.reshape(512, 512)

    return labels_output


def _create_row_column_features_for_predict(predictions: np.array):
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


def _prepare_data_for_predict(features_tensor: np.array, preprocess_info: dict):
    """ Prepare pytorch tensors """
    # Normalize data
    features_tensor = _apply_normalization(np.expand_dims(features_tensor, axis=0),
                                           preprocess_info)
    features_tensor = np.array(features_tensor, dtype='float32')

    features_tensor = torch.from_numpy(features_tensor)
    return features_tensor


def _apply_normalization(features_tensor: np.array, preprocessing_info: dict):
    """ Perform normalization procedure for new unseen data """
    n_objects, n_bands, n_rows, n_columns = features_tensor.shape

    for band_id in range(n_bands):
        band_tensor = features_tensor[:, band_id, :, :]
        min_value = preprocessing_info['info'][str(band_id)]['min']
        max_value = preprocessing_info['info'][str(band_id)]['max']

        features_tensor[:, band_id, :, :] = cv.normalize(band_tensor, min_value, max_value)

    return features_tensor


def _load_neural_networks() -> dict:
    """ Load PyTorch neural networks from serialized path """
    serialized_path = str(SERIALIZED_MODELS_DIR)
    files = os.listdir(serialized_path)
    networks_names = [file for file in files if file.endswith('.pth')]

    models = {}
    for network_name in networks_names:
        model_path = os.path.join(serialized_path, network_name)
        nn_model = torch.load(model_path)
        models.update({network_name: nn_model})

    return models


def _load_final_model():
    """ Load serialised model for ensembling """
    serialized_path = str(SERIALIZED_MODELS_DIR)
    files = os.listdir(serialized_path)

    final_model = None
    if 'final_model.pkl' in files:
        path = os.path.join(serialized_path, 'final_model.pkl')
        with open(path, "rb") as f:
            final_model = pickle.load(f)
    return final_model


# Load neural networks to give prediction
nn_models = _load_neural_networks()
# And final model for ensembling
final_model = _load_final_model()

for chip_id in chip_ids:
    band_arrs = []
    for band in BANDS:
        band_arr = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}/{band}.tif")
        band_arrs.append(band_arr)
    chip_arr = np.stack(band_arrs)

    # Get forecast from the system of several neural networks
    prediction = make_composite_prediction(chip_arr, nn_models, final_model)

    prediction = np.array(prediction, dtype="uint8")
    output_path = PREDICTIONS_DIRECTORY / f"{chip_id}.tif"
    imsave(output_path, prediction)