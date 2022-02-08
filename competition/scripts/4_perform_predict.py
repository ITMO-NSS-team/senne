from pathlib import Path
import numpy as np
from tifffile import imsave, imread

from senne.blending.ml import MLEnsemble
from senne.senne import load_json_files

ROOT_DIRECTORY = Path("..")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
INPUT_IMAGES_DIRECTORY = ROOT_DIRECTORY / "data/train_features"
SERIALIZED_MODELS_DIR = ROOT_DIRECTORY / "serialized"

BANDS = ["B02", "B03", "B04", "B08"]
DEVICE = 'cuda'
chip_ids = (pth.name for pth in INPUT_IMAGES_DIRECTORY.iterdir() if not pth.name.startswith("."))

#################
# PREPARE MODEL #
#################
metadata_path = ROOT_DIRECTORY / "data/train_metadata.csv"
boundaries_info, networks_info = load_json_files(SERIALIZED_MODELS_DIR)
ensemble = MLEnsemble(boundaries_info=boundaries_info, networks_info=networks_info,
                      path=SERIALIZED_MODELS_DIR, metadata_path=None,
                      device=DEVICE, for_predict=True, model_name='naive_bayes',
                      use_shift=True)
#################
# PREPARE MODEL #
#################

for chip_id in chip_ids:
    band_arrs = []
    for band in BANDS:
        band_arr = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}/{band}.tif")
        band_arrs.append(band_arr)
    chip_arr = np.stack(band_arrs)

    # Get forecast from the system of several neural networks
    prediction = ensemble.predict(chip_arr)

    output_path = PREDICTIONS_DIRECTORY / f"{chip_id}.tif"
    imsave(output_path, prediction)
