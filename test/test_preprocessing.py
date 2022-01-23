import numpy as np

from senne.data.preprocessing import apply_normalization


def get_test_matrix():
    """ Generate matrix for tests and transformation info """
    features_matrix = np.array([[0, 11, 10, 11],
                                [11, 10, 11, 10],
                                [10, 11, 10, 11],
                                [11, 10, 11, 10]], dtype=int)

    preprocessing_info = {'preprocessing_name': 'default',
                          'info': {'0': {'min': 0, 'max': 11}}}
    return np.expand_dims(features_matrix, axis=0), preprocessing_info


def test_normalization_correct():
    """ Check if normalization for matrices perform correctly """
    tensor, preprocessing_info = get_test_matrix()
    scaled_tensor = apply_normalization(tensor, preprocessing_info)

    assert np.isclose(scaled_tensor[0, 0, 0, 0], -0.5)
    assert np.isclose(scaled_tensor[0, 0, 0, 1], 0.5)
    assert np.isclose(scaled_tensor[0, 0, 0, 2], 0.40909094)
