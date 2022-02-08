import json
import os
import pandas as pd


def check_time_locations(metadata_path, df_to_test):
    metadata = pd.read_csv(metadata_path, parse_dates=['datetime'])
    metadata['datetime'] = metadata['datetime'].dt.month
    test_df = pd.read_csv(df_to_test)

    time_locations = metadata['datetime'].unique()
    print(f'Number of time locations: {len(time_locations)}')

    # Take chips names
    chip_names = []
    for target_path in test_df['target']:
        splitted_path = os.path.split(target_path)
        name_components = splitted_path[-1].split('.')
        chip_names.append(name_components[0])
    test_df['chip_id'] = chip_names

    merged = metadata.merge(test_df, on='chip_id')
    merged_locations = merged['datetime'].unique()
    print(f'Number of time locations after merging: {len(merged_locations)}')

    locations = set(time_locations)
    merged_locations = set(merged_locations)

    non_used_locations = list(locations - merged_locations)
    print(non_used_locations)
    return non_used_locations


def update_weights(non_used_locations: list, path_to_json_file: str, common_weights: dict):
    # Define number of models
    nn_number = len([key for key in common_weights.keys() if 'th_' in key])

    print(f'Number of neural networks {nn_number}')
    with open(path_to_json_file) as json_file:
        weights = json.load(json_file)

    for parameter in weights.keys():
        for non_used_location in non_used_locations:
            if non_used_location in parameter:
                # There is a need to replace weights
                for nn_id in range(nn_number):
                    if f'th_{nn_id}' in parameter:
                        # Replace threshold
                        new_value = common_weights.get(f'th_{nn_id}')
                        weights.update({parameter: new_value})
                    elif f'weight_{nn_id}' in parameter:
                        new_value = common_weights.get(f'weight_{nn_id}')
                        weights.update({parameter: new_value})

    with open(path_to_json_file, 'w') as f:
        json.dump(weights, f)


if __name__ == '__main__':
    metadata_path = '../data/train_metadata.csv'
    df_to_test = '../serialized/test.csv'
    non_used_locations = check_time_locations(metadata_path, df_to_test)

    # Update non optimized location weights
    # json_path = 'D:/ITMO/sub/serialized/ensemble_info.json'
    # update_weights(non_used_locations, json_path, {"th_0": 0.4331275203245554, "th_1": 0.19828315551394424,
    #                                                "weight_0": 0.38967481822432576, "weight_1": 0.09922608313550485})

