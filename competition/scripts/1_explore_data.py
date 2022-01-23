from senne.data.data import DataProcessor

if __name__ == '__main__':
    # Example of usage data loader to display plots
    # 1 - cloud / 0 - no cloud
    data_processor = DataProcessor(features_path='../data_ensemble/train_features',
                                   target_path='../data_ensemble/train_labels')
    data_processor.explore()
