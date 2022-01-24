from senne.senne import Ensembler

if __name__ == '__main__':
    # An example of how to launch ensembling algorithm for already trained neural networks
    ensembler = Ensembler(path='serialized_folder', device='cuda')

    # Create an ensemble
    ensembler.prepare_composite_model(data_paths={'features_path': '../data_ensemble/train_features',
                                                  'target_path': '../data_ensemble/train_labels'},
                                      final_model='logit',
                                      sampling_ratio=0.05)
