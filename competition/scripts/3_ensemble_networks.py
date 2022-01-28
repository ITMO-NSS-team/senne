from senne.senne import Ensembler

if __name__ == '__main__':
    # An example of how to launch ensembling algorithm for already trained neural networks
    ensembler = Ensembler(path='../sample_serialized', device='cuda')

    # Create an ensemble
    ensembler.prepare_composite_model(data_paths={'features_path': '../data/train_features',
                                                  'target_path': '../data/train_labels'},
                                      final_model='weighted',
                                      sampling_ratio=0.05)
