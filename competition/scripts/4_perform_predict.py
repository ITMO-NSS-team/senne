from senne.predict import MatrixPredict

if __name__ == '__main__':
    predict_creator = MatrixPredict(data_paths={'features_path': '../data_simple/train_features',
                                                'target_path': '../data_simple/train_labels'},
                                    serialized_path='example_folder', final_model='logit',
                                    device='cuda')

    predict_creator.make_prediction(vis=True, take_first_images=50,
                                    save_folder='saved_predictions')