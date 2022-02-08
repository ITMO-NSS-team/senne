import json
import pandas as pd
from pathlib import Path
import segmentation_models_pytorch as smp
import numpy as np
from tifffile import imread
from typing import Callable
import timeit

from senne.blending.ml import MLEnsemble
from senne.data.data import DataProcessor
from senne.senne import load_json_files, torch
from senne.blending.weighted import WeightedEnsemble

ROOT_DIRECTORY = Path("..")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
SERIALIZED_MODELS_DIR = ROOT_DIRECTORY / "serialized"
test_paths = SERIALIZED_MODELS_DIR / "test.csv"

BANDS = ["B02", "B03", "B04", "B08"]
DEVICE = 'cuda'
test_paths_df = pd.read_csv(test_paths)
metadata_path = ROOT_DIRECTORY / "data/train_metadata.csv"


def configure_weighted_model():
    """ Load serializes weighted model """
    boundaries_info, networks_info = load_json_files(SERIALIZED_MODELS_DIR)
    ensemble = WeightedEnsemble(boundaries_info=boundaries_info,
                                networks_info=networks_info,
                                path=SERIALIZED_MODELS_DIR,
                                metadata_path=None,
                                device=DEVICE, for_predict=True)
    # Load coefficients
    path_to_json_file = SERIALIZED_MODELS_DIR / 'ensemble_info.json'
    with open(path_to_json_file) as json_file:
        weights = json.load(json_file)
    ensemble.parameters = weights

    return ensemble


def configure_ml_model():
    boundaries_info, networks_info = load_json_files(SERIALIZED_MODELS_DIR)
    ensemble = MLEnsemble(boundaries_info=boundaries_info,
                          networks_info=networks_info,
                          path=SERIALIZED_MODELS_DIR,
                          metadata_path=None,
                          device=DEVICE, for_predict=True,
                          model_name='rf', use_shift=True)

    return ensemble


def calculate_for_ensemble_model(loading_function: Callable):
    """ How to calculate predictions errors for weighted ensemble """
    ensemble = loading_function()

    metrics = []
    for row_id in range(0, len(test_paths_df)):
        current_row = test_paths_df.iloc[row_id]
        feature_path = current_row['feature']
        target_path = current_row['target']

        chip_arr = DataProcessor.read_geotiff_file(feature_path)

        # Get forecast from the system of several neural networks
        prediction = ensemble.predict(chip_arr)

        actual_mask = imread(target_path)

        prediction = torch.from_numpy(prediction)
        actual_mask = torch.from_numpy(actual_mask)
        iou_metric = smp.utils.metrics.IoU()
        calculated_metric = iou_metric.forward(prediction, actual_mask)
        metrics.append(float(calculated_metric))

    test_paths_df['IoU'] = metrics
    print(f'Calculated IoU metric: {np.mean(np.array(metrics)):.2f}')
    test_paths_df.to_csv('calculated_metrics.csv', index=False)


if __name__ == '__main__':
    start = timeit.default_timer()
    calculate_for_ensemble_model(configure_ml_model)
    print(timeit.default_timer()-start)
