from senne.data.data import SenneDataLoader

if __name__ == '__main__':
    # Example of usage data loader to display plots
    data_processor = SenneDataLoader(features_path='../data_ensemble/train_features',
                                     target_path='../data_ensemble/train_labels')
    data_processor.explore()
