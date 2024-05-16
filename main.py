import os
import json
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter
from settings import Paths, Settings
from dataset import PilotEEGDataLoader
from data_preprocess import DataPreprocessor
from feature_extraction import FeatureExtractor
from models import train_xgb, train_ldgd, train_fast_ldgd
from utils import *


# Load settings from settings.json
settings = Settings()  # Initialize settings object
settings.load_settings()  # Load settings from a JSON file

# Set up paths for data
paths = Paths(settings)  # Initialize paths object with loaded settings
paths.load_device_paths()  # Load device-specific paths
paths.create_paths()  # Create any necessary file paths

# Check if features are available
if os.path.exists(paths.feature_file_path):
    features_raw_df = pd.read_csv(paths.feature_file_path)
else:
    # Load EEG dataset using configured settings and paths
    dataset = PilotEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
    dataset.load_data(patient_ids=settings.patient)  # Load EEG data for specified patients

    # Preprocess the loaded dataset
    data_preprocessor = DataPreprocessor(paths=paths, settings=settings)  # Initialize data preprocessor
    dataset = data_preprocessor.preprocess(dataset)  # Apply preprocessing steps to the dataset

    # Extract features from the preprocessed dataset
    feature_extractor = FeatureExtractor(paths=paths, settings=settings)  # Initialize feature extractor
    feature_extractor.extract_features(dataset)  # Extract relevant features from the dataset
    features_raw_df, *_ = feature_extractor.get_feature_array(dataset)

# Define the columns to drop from the dataset
drop_columns = ['id', 'subject_file', 'block_number', 'block_type', 'stim_indicator', 'go_nogo', 'is_experienced',
                'is_resp', 'is_correct', 'stim']


result_list = {
    'subject id': [],
    'filename': []
}

for method in settings.classifier_list:
    result_list[method + ' avg_f1_score'] = []
    result_list[method + ' std_f1_score'] = []
    result_list[method + ' avg_accuracy'] = []
    result_list[method + ' std_accuracy'] = []
    result_list[method + ' avg_recall'] = []
    result_list[method + ' std_recall'] = []
    result_list[method + ' avg_precision'] = []
    result_list[method + ' std_precision'] = []

# Define the KFold cross-validator
if isinstance(settings.cross_validation_mode, int):
    kf = KFold(n_splits=int, shuffle=True, random_state=42)
elif isinstance(settings.cross_validation_mode, str) and settings.cross_validation_mode == 'block':
    kf = None
else:
    raise ValueError("cv should be number of folds or be 'block' for block based")


patients = list(np.unique(features_raw_df['id']))

certain = []
for patient_id in patients:
    print(f"============ Subject {patient_id} from {len(patients)} ============ \n")
    # select patient
    columns_to_remove = [col for col in features_raw_df.columns if "EX" in col]
    features_raw_df = features_raw_df.drop(columns=columns_to_remove)
    features_df = features_raw_df[features_raw_df['id'] == patient_id]
    # features_df = features_raw_df

    # select labels
    y_one_hot, labels_array, unique_pids, patients_files, features_df = get_labels(features_df, settings)

    result_list['subject id'].append(unique_pids)
    result_list['filename'].append(patients_files.split('_')[0])

    paths.create_subject_paths(patients_files)

    # Perform cross-validation
    fold_acc, fold_f1_score = {method: [] for method in settings.classifier_list}, {method: [] for method in settings.classifier_list}
    fold_precision, fold_recall = {method: [] for method in settings.classifier_list}, {method: [] for method in settings.classifier_list}

    if kf is None:
        block_nums = features_df['block_number'].unique()
        folds = [(np.where(features_df['block_number'] != block)[0],
                  np.where(features_df['block_number'] == block)[0]) for block in block_nums]

        fold_blocks = [(features_df[features_df['block_number'] != block]['block_number'].unique(),
                  features_df[features_df['block_number'] == block]['block_number'].unique()) for block in block_nums]

    else:
        folds = kf.split(features_df)

    folds_info = {
        'train blocks': [],
        'train blocks type': [],
        'test blocks': [],
        'test blocks type': [],
        'number of train_samples (Experienced, Not experienced)': [],
        'number of test_samples (Experienced, Not experienced)': [],
        'Label imbalance (Experienced/Not experienced) train': [],
        'Label imbalance (Experienced/Not experienced) test': []

    }
    for fold_idx, (train_index, test_index) in enumerate(folds):
        paths.create_fold_path(fold_idx)
        train_blocks = list(features_df.iloc[np.array(train_index)]['block_number'].unique())
        test_blocks = list(features_df.iloc[np.array(test_index)]['block_number'].unique())
        train_blocks_type = [features_df[features_df['block_number'] == tb]['block_type'].values[0] for tb in train_blocks]
        test_blocks_type = [features_df[features_df['block_number'] == tb]['block_type'].values[0] for tb in test_blocks]
        experienced_train = features_df.iloc[np.array(train_index)]['is_experienced']
        experienced_test = features_df.iloc[np.array(test_index)]['is_experienced']

        folds_info['train blocks'].append(train_blocks)
        folds_info['train blocks type'].append(train_blocks_type)
        folds_info['test blocks'].append(test_blocks)
        folds_info['test blocks type'].append(test_blocks_type)
        folds_info['number of train_samples (Experienced, Not experienced)'].append((experienced_train.sum(), np.sum(1-experienced_train)))
        folds_info['number of test_samples (Experienced, Not experienced)'].append((experienced_test.sum(), np.sum(1-experienced_test)))
        folds_info['Label imbalance (Experienced/Not experienced) train'].append(experienced_train.sum()/np.sum(1-experienced_train))
        folds_info['Label imbalance (Experienced/Not experienced) test'].append(experienced_test.sum()/ np.sum(1-experienced_test))

        print(f"Training on blocks {train_blocks}({train_blocks_type}) and testing on block {test_blocks}({test_blocks_type})")
        print(f"Number of experienced label in training set: {experienced_train.sum()} "
              f"and in test set {experienced_test.sum()}")
        print(f"Number of not-experienced label in training set: {np.sum(1-experienced_train)} and in test set {np.sum(1-experienced_test)}")
        print(f"Class imbalanced ratio (experienced/not-experinced) in "
              f"train is {np.round(experienced_train.sum()/np.sum(1-experienced_train),2)} and "
              f"in test is {np.round(experienced_test.sum()/np.sum(np.sum(1-experienced_test)),2)}")

        # select features
        features_matrix, selected_features, patients_ids, patients_files = \
            get_selected_features(features_df.copy(), settings, paths,
                                  fold_idx, train_index, train_index,
                                  target_columns_drop=drop_columns)

        data_train, data_test = features_matrix[train_index], features_matrix[test_index]
        labels_train, labels_test = labels_array[train_index], labels_array[test_index]
        y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]
        pid_train, pid_test = patients_ids[train_index], patients_ids[test_index]

        for method in settings.classifier_list:
            if method.lower() == 'xgboost':
                results = train_xgb(data_train, labels_train, data_test, labels_test, paths)
            elif method.lower() == 'ldgd':
                results = train_ldgd(data_train, labels_train, data_test, labels_test,
                                     y_train, y_test,
                                     settings, paths,
                                     shared_inducing_points=False,
                                     use_shared_kernel=False,
                                     cls_weight=settings.cls_weight,
                                     reg_weight=1.0,
                                     early_stop=None)
            elif method.lower() == 'fast_ldgd':
                results = train_fast_ldgd(data_train, labels_train, data_test, labels_test,
                                     y_train, y_test,
                                     settings, paths,
                                     shared_inducing_points=False,
                                     use_shared_kernel=False,
                                     cls_weight=settings.cls_weight,
                                     reg_weight=1.0,
                                     early_stop=None)
            else:
                raise ValueError("Method should be 'xgboost' or 'ldgd'")

            fold_acc[method].append(results['accuracy'])
            fold_f1_score[method].append(results['f1_score'])
            fold_precision[method].append(results['precision'])
            fold_recall[method].append(results['recall'])

        plt.close('all')
    pd.DataFrame(folds_info).to_csv(f"fold_info_{patients_files.split('_')[0]}.csv")
    # Compute average scores
    for method in method_list:
        # Compute average scores
        avg_accuracy = np.mean(fold_acc[method])
        avg_f1_score = np.mean(fold_f1_score[method])
        avg_precision = np.mean(fold_precision[method])
        avg_recall = np.mean(fold_recall[method])

        std_accuracy = np.std(fold_acc[method])
        std_f1_score = np.std(fold_f1_score[method])
        std_precision = np.std(fold_precision[method])
        std_recall = np.std(fold_recall[method])

        print(
            f"Method {method}: f1-score: {avg_f1_score} +- {std_f1_score}  \t accuracy: {avg_accuracy} += {std_accuracy}")

        result_list[method + ' avg_f1_score'].append(avg_f1_score)
        result_list[method + ' avg_accuracy'].append(avg_accuracy)
        result_list[method + ' avg_precision'].append(avg_precision)
        result_list[method + ' avg_recall'].append(avg_recall)

        result_list[method + ' std_f1_score'].append(std_f1_score)
        result_list[method + ' std_accuracy'].append(std_accuracy)
        result_list[method + ' std_precision'].append(std_precision)
        result_list[method + ' std_recall'].append(std_recall)

try:
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(paths.base_path + paths.folder_name + '\\results.csv')
except:
    with open(paths.base_path + paths.folder_name + '\\results_all.json', 'w') as file:
        json.dump(result_list, file, default=convert, indent=2)

try:
    # Flatten the list of lists
    flat_list = list(itertools.chain(*result_list['used features']))

    # Count the frequency of each item
    # Count the frequency of each item
    item_counts = Counter(flat_list)

    # Sort items by count and select the top 25
    top_25_items = item_counts.most_common(25)

    # Separate the items and their counts for plotting
    items, counts = zip(*top_25_items)

    # Plotting
    plt.figure(figsize=(10, 8))  # Adjust the size as needed
    plt.bar(items, counts)
    plt.xticks(rotation=90)  # Rotate labels to make them readable
    plt.xlabel('Feature Names')
    plt.ylabel('Frequency')
    plt.title('Top 25 Feature Names by Frequency')
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.savefig(paths.base_path + paths.folder_name + '\\hist.png')
except:
    pass
