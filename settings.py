import json
import datetime
from pathlib import Path
import configparser
import os
import ast


class Settings:
    def __init__(self, patient='all', verbose=True):
        self.__load_trained_model = False
        self.__target_column = 'is_experienced'
        self.__features_selection_method = 'all'  # Default value
        self.__classifier_list = ['fast_ldgd', 'ldgd']  # Default value
        self.__supported_feature_transformation = ['normalize', 'standardize', 'none']
        self.__supported_datasets = ['pilot01']
        self.__supported_classifiers = ['ldgd', 'fast_ldgd', 'xgboost']

        self.patient = patient
        self.verbose = verbose

        self.__debug_mode = False
        self.__save_features = False
        self.__load_pretrained_model = False
        self.__num_fold = 5
        self.__test_size = 0.2
        self.__load_features = False
        self.__feature_transformation = 'normalize'  # Default value
        self.__dataset = 'pilot01'  # Default value

        self.__load_epoched_data = False
        self.__save_epoched_data = False
        self.__load_preprocessed_data = False
        self.__save_preprocessed_data = False

        self.__batch_size = 300
        self.__num_epochs_train = 2000
        self.__num_epochs_test = 2000
        self.__cross_validation_mode = 'block'

        self.__data_dim = None
        self.__latent_dim = 7
        self.__num_inducing_points = 15
        self.__cls_weight = 1
        self.__use_gpytorch = True

    def _remove_comments(self, value):
        if isinstance(value, str) and '#' in value:
            return value.split('#')[0].strip()
        return value.strip()

    def load_settings(self):
        """
        This function loads the ini file for settings from the working directory and
        creates a Settings object based on the fields in the ini file.
        """
        working_folder = os.path.abspath(os.path.dirname(__file__))
        parent_folder = working_folder

        """ loading settings from the ini file """
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(parent_folder, "configs/settings.ini"))
        except:
            raise Exception('Could not load settings.ini from the working directory!')

        """ creating settings """
        if 'DEFAULT' not in config:
            raise Exception('"DEFAULT" section was not found in settings.ini!')

        self.patient = self._remove_comments(config['DEFAULT'].get('patient', 'all'))
        self.__dataset = self._remove_comments(config['DEFAULT'].get('dataset', None))

        sections = ['classifier', 'data', 'ldgd', 'training']
        for section in sections:
            if section in config:
                for key, value in config[section].items():
                    clean_value = self._remove_comments(value)
                    if hasattr(self, key):
                        attr = getattr(self, key)
                        if isinstance(attr, bool):
                            setattr(self, key, config.getboolean(section, key))
                        elif isinstance(attr, int):
                            setattr(self, key, config.getint(section, key))
                        elif isinstance(attr, float):
                            setattr(self, key, config.getfloat(section, key))
                        elif key == 'classifier_list':
                            setattr(self, key, ast.literal_eval(clean_value))
                        else:
                            setattr(self, key, clean_value)
                    else:
                        raise Exception(f'{key} is not an attribute of the Settings class!')

        if self.verbose:
            print(f"Patient: {self.patient}")

    @property
    def load_trained_model(self):
        return self.__load_trained_model
    @load_trained_model.setter
    def load_trained_model(self, value):
        if isinstance(value, bool):
            self.__load_trained_model = value
        else:
            raise ValueError("load_trained_model should be True or False")

    @property
    def target_column(self):
        return self.__target_column
    @target_column.setter
    def target_column(self, value):
        self.__target_column = value

    @property
    def features_selection_method(self):
        return self.__features_selection_method
    @features_selection_method.setter
    def features_selection_method(self, value):
        self.__features_selection_method = value

    @property
    def classifier_list(self):
        return self.__classifier_list

    @classifier_list.setter
    def classifier_list(self, value):
        if isinstance(value, list):
            self.__classifier_list = value
        else:
            raise ValueError("classifier_list should be a list of classifiers")

    @property
    def load_epoched_data(self):
        return self.__load_epoched_data

    @load_epoched_data.setter
    def load_epoched_data(self, value):
        if isinstance(value, bool):
            self.__load_epoched_data = value
        else:
            raise ValueError("load_epoched_data should be True or False")

    @property
    def save_epoched_data(self):
        return self.__save_epoched_data

    @save_epoched_data.setter
    def save_epoched_data(self, value):
        if isinstance(value, bool):
            self.__save_epoched_data = value
        else:
            raise ValueError("save_epoched_data should be True or False")

    @property
    def load_preprocessed_data(self):
        return self.__load_preprocessed_data

    @load_preprocessed_data.setter
    def load_preprocessed_data(self, value):
        if isinstance(value, bool):
            self.__load_preprocessed_data = value
        else:
            raise ValueError("load_preprocessed_data should be True or False")

    @property
    def save_preprocessed_data(self):
        return self.__save_preprocessed_data

    @save_preprocessed_data.setter
    def save_preprocessed_data(self, value):
        if isinstance(value, bool):
            self.__save_preprocessed_data = value
        else:
            raise ValueError("save_preprocessed_data should be True or False")

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset_name):
        if isinstance(dataset_name, str) and dataset_name in self.__supported_datasets:
            self.__dataset = dataset_name
        else:
            raise ValueError(f"The dataset should be selected from {self.__supported_datasets}")

    @property
    def feature_transformation(self):
        return self.__feature_transformation

    @feature_transformation.setter
    def feature_transformation(self, value):
        if value is None:
            self.__feature_transformation = None
        elif isinstance(value, str) and value.lower() in self.__supported_feature_transformation:
            self.__feature_transformation = value.lower()
        else:
            raise ValueError(
                f"The feature_transformation should be selected from {self.__supported_feature_transformation}")

    @property
    def load_features(self):
        return self.__load_features

    @load_features.setter
    def load_features(self, value):
        if isinstance(value, bool):
            self.__load_features = value
        else:
            raise ValueError("load_features should be True or False")

    @property
    def num_fold(self):
        return self.__num_fold

    @num_fold.setter
    def num_fold(self, k):
        if isinstance(k, int) and k > 0:
            self.__num_fold = k
        else:
            raise ValueError("num_fold should be an integer greater than 0")

    @property
    def test_size(self):
        return self.__test_size

    @test_size.setter
    def test_size(self, value):
        if 0 < value < 1:
            self.__test_size = value
        else:
            raise ValueError("test_size should be a float number between 0 and 1")

    @property
    def load_pretrained_model(self):
        return self.__load_pretrained_model

    @load_pretrained_model.setter
    def load_pretrained_model(self, value):
        if isinstance(value, bool):
            self.__load_pretrained_model = value
        else:
            raise ValueError("load_pretrained_model should be True or False")

    @property
    def save_features(self):
        return self.__save_features

    @save_features.setter
    def save_features(self, value):
        if isinstance(value, bool):
            self.__save_features = value
        else:
            raise ValueError("save_features should be True or False")

    @property
    def debug_mode(self):
        return self.__debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        if isinstance(value, bool):
            self.__debug_mode = value
        else:
            raise ValueError("debug_mode should be True or False")

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        if isinstance(value, int) and value > 0:
            self.__batch_size = value
        else:
            raise ValueError("batch_size should be a positive integer")

    @property
    def num_epochs_train(self):
        return self.__num_epochs_train

    @num_epochs_train.setter
    def num_epochs_train(self, value):
        if isinstance(value, int) and value > 0:
            self.__num_epochs_train = value
        else:
            raise ValueError("num_epochs_train should be a positive integer")

    @property
    def num_epochs_test(self):
        return self.__num_epochs_test

    @num_epochs_test.setter
    def num_epochs_test(self, value):
        if isinstance(value, int) and value > 0:
            self.__num_epochs_test = value
        else:
            raise ValueError("num_epochs_test should be a positive integer")

    @property
    def cross_validation_mode(self):
        return self.__cross_validation_mode

    @cross_validation_mode.setter
    def cross_validation_mode(self, value):
        if isinstance(value, str):
            self.__cross_validation_mode = value
        else:
            raise ValueError("cross_validation_mode should be a string")

    @property
    def data_dim(self):
        return self.__data_dim

    @data_dim.setter
    def data_dim(self, value):
        if value is None or isinstance(value, int):
            self.__data_dim = value
        elif isinstance(value, str):
            if value.lower() == 'none':
                self.__data_dim = None
            else:
                raise ValueError("data_dim should be 'None' or an integer")
        else:
            raise ValueError("data_dim should be an integer or None")

    @property
    def latent_dim(self):
        return self.__latent_dim

    @latent_dim.setter
    def latent_dim(self, value):
        if isinstance(value, int) and value > 0:
            self.__latent_dim = value
        else:
            raise ValueError("latent_dim should be a positive integer")

    @property
    def num_inducing_points(self):
        return self.__num_inducing_points

    @num_inducing_points.setter
    def num_inducing_points(self, value):
        if isinstance(value, int) and value > 0:
            self.__num_inducing_points = value
        else:
            raise ValueError("num_inducing_points should be a positive integer")

    @property
    def cls_weight(self):
        return self.__cls_weight

    @cls_weight.setter
    def cls_weight(self, value):
        if isinstance(value, int) and value > 0:
            self.__cls_weight = value
        else:
            raise ValueError("cls_weight should be a positive integer")

    @property
    def use_gpytorch(self):
        return self.__use_gpytorch

    @use_gpytorch.setter
    def use_gpytorch(self, value):
        if isinstance(value, bool):
            self.__use_gpytorch = value
        else:
            raise ValueError("use_gpytorch should be True or False")

class Paths:
    def __init__(self, settings):
        self.base_path = None
        self.path_model = None
        self.path_result = None

        self.xdf_directory_path = './'
        self.feature_file_path = './'
        self.feature_path = './'
        self.path_subject_result = {}
        self.channel_group_file = './data/channel_groups.mat'

        self.patient = settings.patient
        self.debug_mode = settings.debug_mode

    def load_device_paths(self):
        """ working directory """
        working_folder = os.path.abspath(__file__)
        if 'TOI1' in working_folder:
            # Find the index of 'Suspicious_Message_Detection'
            index = working_folder.find('TOI1') + len('TOI1')
            # Extract the path up to 'Suspicious_Message_Detection'
            parent_folder = working_folder[:index]
        else:
            print("The path does not contain 'TOI1'")

        """ loading device path from the ini file """
        config = configparser.ConfigParser()
        try:
            config.read(parent_folder + "/configs/device_path.ini")
        except:
            raise Exception('Could not load device_path.ini from the working directory!')

        device = config['paths']

        for key, value in device.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Paths class!'.format(key))

    def create_paths(self):
        working_folder = os.path.abspath(__file__)
        if 'TOI1' in working_folder:
            # Find the index of 'Suspicious_Message_Detection'
            index = working_folder.find('TOI1') + len('TOI1')
            # Extract the path up to 'Suspicious_Message_Detection'
            dir_path = working_folder[:index]
        else:
            print("The path does not contain 'TOI1'")

        self.base_path = os.path.join(dir_path, 'results', '')
        if not self.debug_mode:
            self.folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.folder_name = 'debug'
        results_base_path = os.path.join(self.base_path, self.folder_name, '')
        """if Path(results_base_path).is_dir():
            shutil.rmtree(results_base_path)"""

        Path(results_base_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(results_base_path, 'model', '')).mkdir(parents=True, exist_ok=True)
        self.path_model = os.path.join(results_base_path, 'model', '')
        self.path_result = results_base_path

    def create_subject_paths(self, subject_name):
        self.results_base_path = os.path.join(self.base_path, self.folder_name, f'{subject_name}', '')
        Path(self.results_base_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.results_base_path, 'model', '')).mkdir(parents=True, exist_ok=True)
        self.path_model = os.path.join(self.results_base_path, 'model', '')
        self.path_result = self.results_base_path

    def create_fold_path(self, fold):
        self.fold_path = os.path.join(self.results_base_path, f'fold{fold}', '')
        Path(self.fold_path).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.fold_path, 'model', '')).mkdir(parents=True, exist_ok=True)
        self.path_model = os.path.join(self.fold_path, 'model', '')
        self.path_result = self.fold_path

    def create_paths_subject(self, patient_id):
        self.path_subject_result[patient_id] = os.path.join(self.path_result, patient_id, '')
        Path(self.path_subject_result[patient_id]).mkdir(parents=True, exist_ok=True)

    def update_path(self, time_index):
        self.path_model_updated, self.path_result_updated = [], []
        for idx in range(len(self.path_result)):
            self.path_model_updated.append(os.path.join(self.path_result[idx], f't_{time_index}', 'model', ''))
            self.path_result_updated.append(os.path.join(self.path_result[idx], f't_{time_index}', ''))
            Path(self.path_model_updated[idx]).mkdir(parents=True, exist_ok=True)
            Path(self.path_result_updated[idx]).mkdir(parents=True, exist_ok=True)