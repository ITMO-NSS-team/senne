from senne.data.data import DataProcessor

if __name__ == '__main__':
    # Example of usage data loader to display plots
    # 1 - cloud / 0 - no cloud
    data_processor = DataProcessor(features_path='../data/train_features',
                                   target_path='../data/train_labels')

    # Find incorrect objects
    data_processor.check_bands_number(correct_band_number=4)
    data_processor.explore()
