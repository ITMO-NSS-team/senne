import os
import numpy as np
from tifffile import imsave, imread

from senne.data.data import create_matrix_plot, DataProcessor

BANDS = ["B02", "B03", "B04", "B08"]


def filter_matrices(path, path_to_save):
    """ Matrices filtering perform manually """
    new_save_train_features = os.path.join(path_to_save, 'train_features')
    new_save_train_labels = os.path.join(path_to_save, 'train_labels')
    os.makedirs(new_save_train_labels)

    train_features = os.path.join(path, 'train_features')
    train_labels = os.path.join(path, 'train_labels')

    areas = os.listdir(train_features)
    for area in areas:
        area_path = os.path.join(train_features, area)
        new_save_area_path = os.path.join(new_save_train_features, area)

        chip_ids = os.listdir(area_path)

        band_arrs = []
        for chip_id in chip_ids:
            band_arr = imread(os.path.join(area_path, chip_id))
            band_arrs.append(band_arr)
        chip_arr = np.stack(band_arrs)

        # Label matrix
        label_matrix_path = os.path.join(train_labels, f'{area}.tif')
        label_matrix = imread(label_matrix_path)

        create_matrix_plot(chip_arr, label_matrix)

        answer = input("1 - correct object, 0 - bad object: ")
        if str(answer) == '1':
            print(f'Good examples {area} was saved')
            os.makedirs(new_save_area_path)
            for i, chip_id in enumerate(chip_ids):
                features_matrix = band_arrs[i]
                imsave(os.path.join(new_save_area_path, chip_id), features_matrix)
                imsave(os.path.join(new_save_train_labels, f'{area}.tif'), label_matrix)
        else:
            print(f'Bad example {area}. Skip it!')


def display_min_max_boundaries():
    data_processor = DataProcessor(features_path='../data/train_features',
                                   target_path='../data/train_labels')
    data_processor.collect_sample_info(serialized_path='new_serialized_folder',
                                       display_labels=True)


if __name__ == '__main__':
    data_path = '../data'
    save_path = '../filtered_data'
    filter_matrices(data_path, save_path)

    display_min_max_boundaries()
