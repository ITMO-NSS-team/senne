import json
import pandas as pd
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tifffile import imread

from senne.blending.ml import MLEnsemble
from senne.data.data import DataProcessor
from senne.senne import load_json_files, torch
from senne.blending.weighted import WeightedEnsemble

ROOT_DIRECTORY = Path("D:/ITMO/sub")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
SERIALIZED_MODELS_DIR = ROOT_DIRECTORY / "serialized"

BANDS = ["B02", "B03", "B04", "B08"]
DEVICE = 'cuda'
test_paths_df = pd.read_csv('../scripts/Calculated_metrics.csv')
metadata_path = ROOT_DIRECTORY / "data/test_metadata.csv"


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
                          model_name='rf')

    return ensemble


if __name__ == '__main__':
    ensemble = configure_ml_model()
    for row_id in range(0, len(test_paths_df)):
        current_row = test_paths_df.iloc[row_id]
        feature_path = current_row['feature']
        target_path = current_row['target']

        chip_arr = DataProcessor.read_geotiff_file(feature_path)

        # Get forecast from the system of several neural networks
        prediction = ensemble.predict(chip_arr)

        actual_mask = imread(target_path)

        prediction_tensor = torch.from_numpy(prediction)
        actual_tensor = torch.from_numpy(actual_mask)
        iou_metric = smp.utils.metrics.IoU()
        calculated_metric = float(iou_metric.forward(prediction_tensor, actual_tensor))
        print(f'{row_id} Metric: {calculated_metric:.4f} for {feature_path[-10:]}')

        fig, axs = plt.subplots(1, 2)
        prediction = np.array(prediction)
        actual_mask = np.array(actual_mask)
        im_first = axs[0].imshow(prediction, cmap='Blues', alpha=1.0, vmin=0, vmax=1)
        divider_first = make_axes_locatable(axs[0])
        cax_first = divider_first.append_axes("right", size="8%", pad=0.2)
        cbar_first = plt.colorbar(im_first, cax=cax_first)
        axs[0].set_title('Ensemble predict')

        im_second = axs[1].imshow(actual_mask, cmap='Blues', alpha=1.0, vmin=0, vmax=1)
        divider_second = make_axes_locatable(axs[1])
        cax_second = divider_second.append_axes("right", size="8%", pad=0.2)
        cbar_second = plt.colorbar(im_second, cax=cax_second)
        axs[1].set_title('Actual matrix')

        plt.tight_layout()
        plt.show()
