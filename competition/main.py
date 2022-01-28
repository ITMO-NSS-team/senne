import pickle
from pathlib import Path
import numpy as np
from tifffile import imsave, imread

ROOT_DIRECTORY = Path("/codeexecution")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
INPUT_IMAGES_DIRECTORY = ROOT_DIRECTORY / "data/test_features"
SERIALIZED_MODELS_DIR = ROOT_DIRECTORY / "serialized"

BANDS = ["B02", "B03", "B04", "B08"]
DEVICE = 'cuda'
chip_ids = (pth.name for pth in INPUT_IMAGES_DIRECTORY.iterdir() if not pth.name.startswith("."))

##############
# LOAD MODEL #
##############
path = SERIALIZED_MODELS_DIR / 'final_model.pkl'
with open(path, "rb") as f:
    ensemble = pickle.load(f)
    ensemble.path = SERIALIZED_MODELS_DIR
    ensemble.device = DEVICE
##############
# LOAD MODEL #
##############

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
