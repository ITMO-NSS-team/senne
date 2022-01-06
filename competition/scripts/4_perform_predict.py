from senne.predict import MatrixPredict

if __name__ == '__main__':
    predict_creator = MatrixPredict(data_paths={'features_path': '../data_simple/train_features',
                                                'target_path': '../data_simple/train_labels'},
                                    serialized_path='example_folder',
                                    device='cuda')

    predict_creator.make_prediction(vis=True)
