from senne.data import SenneDataLoader

data_processor = SenneDataLoader(features_path='../data/train_features',
                                 target_path='../data/train_labels')
data_processor.explore()
