from senne.predict import MatrixPredict

if __name__ == '__main__':
    predict_creator = MatrixPredict(data_paths={'features_path': '../data_ensemble/train_features',
                                                'target_path': '../data_ensemble/train_labels'},
                                    serialized_path='sample_serialized', final_model='weighted',
                                    device='cpu')

    predict_creator.make_prediction(vis=True, take_first_images=50,
                                    save_folder='saved_predictions')