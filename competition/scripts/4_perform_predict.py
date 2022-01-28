import pickle

import numpy as np
import os
from pathlib import Path
from senne.senne import load_json_files
from tifffile import imsave, imread

ROOT_DIRECTORY = Path("..")
INPUT_IMAGES_DIRECTORY = ROOT_DIRECTORY / "data/train_features"
DEVICE = 'cuda'

BANDS = ["B02", "B03", "B04", "B08"]
chip_ids = (pth.name for pth in INPUT_IMAGES_DIRECTORY.iterdir() if not pth.name.startswith("."))


if __name__ == '__main__':
    serialed_path = 'D:/ITMO/senne/competition/serialized'
    path = os.path.join(serialed_path, 'final_model.pkl')
    with open(path, "rb") as f:
        ensemble = pickle.load(f)
        ensemble.path = serialed_path
        ensemble.device = DEVICE

    for chip_id in chip_ids:
        band_arrs = []
        for band in BANDS:
            band_arr = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}/{band}.tif")
            band_arrs.append(band_arr)
        chip_arr = np.stack(band_arrs)
        pr_mask = ensemble.predict(chip_arr)

        import matplotlib.pyplot as plt

        plt.imshow(pr_mask)
        plt.show()
