from senne.senne import Ensembler

if __name__ == '__main__':
    # An example of how to train and fit neural networks before ensembling
    ensembler = Ensembler(path='example_folder', device='cuda')

    # If there is no idea which networks to use - use preset
    preset = 'two_simple'
    ensembler.train_neural_networks(data_paths={'features_path': '../data_simple/train_features',
                                                'target_path': '../data_simple/train_labels'},
                                    preset=preset)
