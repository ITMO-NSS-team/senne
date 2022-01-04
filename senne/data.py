import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

from geotiff import GeoTiff

from senne.log import senne_logger


class SenneDataLoader:
    """ Class for data preprocessing and transformations """

    def __init__(self, features_path: str, target_path: str):
        self.features_path = features_path
        self.target_path = target_path

    def explore(self):
        """ Read and visualize geotiff files in the folder. Display all features matrices and label matrix """
        area_folders = os.listdir(self.features_path)
        for current_area in area_folders:
            senne_logger.info(f'Processing area {current_area}')

            # In area folder there are several geotiff files
            area_path = os.path.join(self.features_path, current_area)
            features_array = self._read_geotiff_file(area_path)

            # Read array with labels
            tiff_label_name = ''.join((current_area, '.tif'))
            opened_label_tiff = GeoTiff(os.path.join(self.target_path, tiff_label_name))
            label_matrix = np.array(opened_label_tiff.read())

            # Create plot
            fig, axs = plt.subplots(2, 2)
            im_first = axs[0, 0].imshow(features_array[0], cmap='Blues', alpha=1.0)
            # Add label matrix
            axs[0, 0].imshow(label_matrix, cmap='Greys', alpha=0.15)
            divider_first = make_axes_locatable(axs[0, 0])
            cax_first = divider_first.append_axes("right", size="8%", pad=0.2)
            cbar_first = plt.colorbar(im_first, cax=cax_first)
            axs[0, 0].set_title('B02')

            im_second = axs[0, 1].imshow(features_array[1], cmap='Greens', alpha=1.0)
            # Add label matrix
            axs[0, 1].imshow(label_matrix, cmap='Greys', alpha=0.15)
            divider_second = make_axes_locatable(axs[0, 1])
            cax_second = divider_second.append_axes("right", size="8%", pad=0.2)
            cbar_second = plt.colorbar(im_second, cax=cax_second)
            axs[0, 1].set_title('B03')

            im_third = axs[1, 0].imshow(features_array[2], cmap='Oranges', alpha=1.0)
            # Add label matrix
            axs[1, 0].imshow(label_matrix, cmap='Greys', alpha=0.15)
            divider_third = make_axes_locatable(axs[1, 0])
            cax_third = divider_third.append_axes("right", size="8%", pad=0.2)
            cbar_third = plt.colorbar(im_third, cax=cax_third)
            axs[1, 0].set_title('B04')

            im_fourth = axs[1, 1].imshow(features_array[3], cmap='Reds', alpha=1.0)
            # Add label matrix
            axs[1, 1].imshow(label_matrix, cmap='Greys', alpha=0.15)
            divider_fourth = make_axes_locatable(axs[1, 1])
            cax_fourth = divider_fourth.append_axes("right", size="8%", pad=0.2)
            cbar_fourth = plt.colorbar(im_fourth, cax=cax_fourth)
            axs[1, 1].set_title('B08')

            plt.tight_layout()
            plt.show()

    @staticmethod
    def _read_geotiff_file(area_path: str) -> np.array:
        """ Read and convert geotiff file as numpy array """
        bands_tiff = os.listdir(area_path)
        bands_tiff.sort()

        final_tensor = []
        for tiff_file in bands_tiff:
            tiff_path = os.path.join(area_path, tiff_file)

            opened_tiff = GeoTiff(tiff_path)
            matrix = np.array(opened_tiff.read())

            final_tensor.append(matrix)

        final_tensor = np.array(final_tensor)
        return final_tensor
