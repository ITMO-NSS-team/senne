from senne.senne import Ensembler
import numpy as np

if __name__ == '__main__':
    # An example of how to launch ensembling algorithm for already trained neural networks
    ensembler = Ensembler(path='example_folder', device='cuda')

    # Create an ensemble
    ensembler.prepare_composite_model(data_paths={'features_path': '../data_simple/train_features',
                                                  'target_path': '../data_simple/train_labels'},
                                      final_model='automl',
                                      sampling_ratio=0.001)
