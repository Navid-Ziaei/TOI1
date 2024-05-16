import pickle

import os
import numpy as np
import scipy.io
import logging
from abc import ABC, abstractmethod
import pyxdf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from visualization import plot_histogram


class AbstractEEGDataLoader(ABC):
    def __init__(self, paths, settings):
        """
        Initialize the data loader with paths and settings.

        Parameters
        ----------
        paths : object
            An object containing various file paths necessary for data loading.
        settings : object
            An object containing various settings for data processing.
        """
        self.fs = None
        self.paths = paths
        self.data_directory = paths.xdf_directory_path
        self.channel_group_file = paths.channel_group_file
        self.history_length = 1
        # Initializations can vary based on subclass implementation
        self.channel_groups = None
        self.all_patient_data = {}

    @abstractmethod
    def load_data(self, patient_ids='all'):
        """
        Load EEG data for specified patients.

        Parameters
        ----------
        patient_ids : str or list of str, optional
            The IDs of the patients whose data is to be loaded.
            If 'all', data for all patients in the directory will be loaded.
        """
        pass

    @abstractmethod
    def load_single_patient_data(self, data, dgd_outputs=None):
        """
        Process and load data for a single patient.

        Parameters
        ----------
        data : dict
            The raw data loaded from the file.
        dgd_outputs : ndarray
            The DGD output data corresponding to the patient.

        Returns
        -------
        Any
            An instance containing the processed data for a single patient.
        """
        pass


class PilotEEGDataLoader(AbstractEEGDataLoader):
    def __init__(self, paths, settings):
        super().__init__(paths, settings)
        self.channel_groups = scipy.io.loadmat(self.channel_group_file)['ChGrp'][0]
        self.settings = settings

    def load_data(self, patient_ids='all'):
        """
        Load EEG data for specified patients.

        This method loads the EEG data and associated DGD outputs for each patient
        and processes it into a format suitable for further analysis.

        Parameters
        ----------
        patient_ids : str or list of str, optional
            The IDs of the patients whose data is to be loaded.
            If 'all', data for all patients in the directory will be loaded.

        Raises
        ------
        ValueError
            If no files are found for the specified patient IDs.
        """
        file_list = os.listdir(self.data_directory)
        if isinstance(patient_ids, str) and patient_ids == 'all':
            xdf_file_list = [file for file in file_list if file.endswith(".xdf")]
        else:
            if isinstance(patient_ids, str):
                patient_ids = [patient_ids]
            xdf_file_list = []
            for patient in patient_ids:
                xdf_file_list.extend([f for f in os.listdir(self.data_directory) if f.endswith(".xdf")
                                      and f.startswith(str(patient))])

            if len(file_list) == 0:
                raise ValueError(f"No patient found with name {patient_ids}")

        for index, file_name in enumerate(xdf_file_list):
            print(f"Subject {index} from {len(xdf_file_list)}: {file_name.split('.')[0]} Load Data")
            logging.info(f"Subject {index} from {len(file_list)}: {file_name.split('.')[0]} Load Data ...")
            # If specific patient IDs are provided, skip files not matching those IDs

            # streams, fileheader = pyxdf.load_xdf(file_paths)
            prepaired_data_path = self.paths.xdf_directory_path + file_name.split('.')[0] + "_trials.pkl"
            if os.path.exists(prepaired_data_path) and self.settings.load_epoched_data is True:
                with open(prepaired_data_path, 'rb') as file:
                    dataset = pickle.load(file)
            else:
                dataset = self.load_single_patient_data(file_name)
                if self.settings.save_epoched_data is True:
                    dataset.save_to_pickle(
                        file_path=self.paths.xdf_directory_path + file_name.split('.')[0] + "_trials.pkl")
            self.all_patient_data[file_name.split('.')[0]] = dataset

    def load_single_patient_data(self, data, dgd_outputs=None, preprocess_continuous_data=False):
        file_name = data

        file_path = os.path.join(self.data_directory, file_name)

        streams, fileheader = pyxdf.load_xdf(file_path)
        for stream_idx in range(len(streams)):
            channel_info = streams[stream_idx]['info']['desc'][0]['channels']
            if len(channel_info) > 0 and len(channel_info[0]['channel']) > 100:
                # Assuming you have one EEG stream, and it's the first stream (index 0)
                eeg_data = streams[stream_idx]['time_series'].T  # Transpose to have channels as rows
                eeg_times = streams[stream_idx]['time_stamps'] - streams[stream_idx]['time_stamps'][0]
                channel_info = streams[stream_idx]['info']['desc'][0]['channels']
                channel_names = [channel['label'][0] for channel in channel_info[0]['channel']]
                subject = streams[stream_idx]['info']['desc'][0]['subject']
                stream_id = streams[stream_idx]['info']['stream_id']
                s_rate = float(streams[stream_idx]['info']['nominal_srate'][0])
                reference_electrode = streams[stream_idx]['info']['desc'][0]['reference'][0]['label']
                subject_id = streams[stream_idx]['info']['desc'][0]['subject'][0]['id'][0]
                if len(streams[stream_idx]['info']['desc'][0]['subject'][0]['group']) > 0:
                    subject_group = streams[stream_idx]['info']['desc'][0]['subject'][0]['group'][0]
                else:
                    subject_group = None

                channel_types = ['eeg' for _ in range(eeg_data.shape[0])]

                print(f" File {file_name} uses stream {stream_idx}")
                break

        fig, axs = plt.subplots(4, 4, figsize=(40, 10))
        for i in range(4):
            for j in range(4):
                axs[i, j].plot(eeg_times[:1000], eeg_data[i + j * 4, :1000])
                axs[i, j].set_title(channel_names[i + j * 4])
                axs[i, j].set_xlabel("Time")
                axs[i, j].set_ylabel("Amp")
        plt.tight_layout()
        plt.show()

        keep_indices = [i for i, name in enumerate(channel_names) if "AUX" not in name and "Trig" not in name]
        # Filter the EEG data to remove channels with "AUX" in their names
        eeg_data = eeg_data[keep_indices, :]

        # Also, filter the channel names list
        channel_names = [name for i, name in enumerate(channel_names) if i in keep_indices]

        if eeg_data.shape[0] != len(channel_names):
            eeg_data = eeg_data.T  # Transpose if necessary

        # Create an Info object
        eeg_data = eeg_data / np.quantile(np.abs(eeg_data), 0.99, axis=-1, keepdims=True)
        eeg_data = eeg_data - np.mean(eeg_data, axis=-1, keepdims=True)


        formatted_marker_df = self._reformat_marker_file(
            marker_path=self.data_directory + file_name.split('.')[
                0] + '.recoded.merged.preproc.curated.result.xdf.markers.csv',
            file_name=file_name, subject_group=subject_group, load_reformatted_data=False)

        eeg_trial_data, eeg_trial_time, eeg_labels, trial_length, trial_index = self._convert_continuous_to_trial(
            eeg_times=eeg_times,
            eeg_data=eeg_data,
            s_rate=s_rate,
            file_name=file_name,
            formatted_marker_df=formatted_marker_df,
            save_data=True,
            load_trialed_data=False)

        plot_histogram(trial_length, xlabel='Length trial (second)', ylabel='Number of Trials',
                       title=f"Histogram of Trial lengths for subject {file_name.split('_')[0]}")

        try:
            bad_channels_df = pd.read_csv(self.data_directory + file_name + '.bad_channels.csv')
            bad_channels = bad_channels_df.columns.to_list()
        except:
            print("N vbadchannel is detected")
            bad_channels = []

        dataset = EEGDataSet()
        dataset.data = eeg_trial_data
        dataset.response_time = np.squeeze(trial_length)
        dataset.labels = eeg_labels
        dataset.trial_index = np.squeeze(trial_index)
        dataset.fs = s_rate
        dataset.time_ms = np.squeeze(np.round(eeg_trial_time * 1000))
        dataset.channel_names = channel_names
        dataset.file_name = file_name
        dataset.stream_id = stream_id
        dataset.reference_electrodes = reference_electrode
        dataset.channel_group = self.channel_groups
        dataset.bad_channels = bad_channels

        return dataset

    def _reformat_marker_file(self, marker_path, file_name, subject_group, load_reformatted_data=True):
        if load_reformatted_data is True:
            formatted_data = pd.read_csv(self.data_directory + file_name.split('.')[0] + '_reformatted.csv')
        else:
            marker_df = pd.read_csv(marker_path)
            column_name = marker_df.columns.to_list()
            if 'event' not in column_name:
                marker_df.rename(columns={column_name[0]: 'event', column_name[1]: 'time'}, inplace=True)

            # Initialize a trial index column
            marker_df['trial_index'] = None

            # Variable to keep track of the current trial index
            current_trial_index = None

            # Initialize a trial index, block index and block type column
            marker_df['trial_index'], marker_df['block_index'], marker_df['block_type'] = None, None, None

            # Variable to keep track of the current trial index
            current_trial_index, current_block_index, current_block_type = None, None, None

            # Iterate through the DataFrame to assign trial index based on "block-begin-x" events
            for index, row in marker_df.iterrows():
                if 'subject_id' in row['event'] and subject_group is None:
                    subject_group = row['event'].split(':')[-1].replace(' ', '')
                if 'trial-begin' in row['event']:
                    # Extract the trial index from the event string
                    current_trial_index = int(row['event'].split('-')[-1])

                if 'block-begin' in row['event']:
                    # Extract the trial index from the event string
                    current_block_index = int(row['event'].split('-')[2])

                    if 'type' in row['event']:
                        current_block_type = row['event'].split('_')[-1]
                        marker_df.at[index, 'block_type'] = current_block_type
                    else:
                        raise ValueError("Can not locate the block type")
                marker_df.at[index, 'trial_index'] = current_trial_index
                marker_df.at[index, 'block_index'] = current_block_index
                marker_df.at[index, 'block_type'] = current_block_type

            marker_df['subject_group'] = subject_group
            marker_df['stim'] = None
            marker_df['stim_indicator'] = None
            marker_df['go_nogo'] = None
            marker_df['exp_label'] = None
            marker_df['is_resp'] = None
            marker_df['is_correct'] = None
            marker_df['response_time'] = None
            marker_df['block_type'] = None
            marker_df['stim_desc'] = None
            for index, row in marker_df.iterrows():
                if 'stim_' in row['event']:
                    stim_type = row['event'].split('_')[1]
                    block_type = row['event'].split('_')[2]
                    task_type = row['event'].split('_')[3]
                    stim_desc = row['event'].split('_')[4]

                    if stim_type in ['stp', 'msp', 'ctl']:
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim'] = stim_type
                    else:
                        raise ValueError(f"Stim type mismatch: {stim_type} in {row['event']}")

                    if block_type in ['w+e', 'w-e', 'w+e+x', 'w-e+x']:
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_indicator'] = 'word'
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'block_type'] = block_type
                    elif block_type in ['i+e', 'i-e', 'i+e+x', 'i-e+x']:
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_indicator'] = 'image'
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'block_type'] = block_type
                    else:
                        raise ValueError(f"Block type type mismatch: {block_type} in {row['event']}")

                    if task_type in ['nogo', 'go']:
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'go_nogo'] = task_type
                    else:
                        raise ValueError(f"task type mismatch: {task_type} in {row['event']}")

                    marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_desc'] = stim_desc

                if 'resp_' in row['event']:
                    is_resp = row['event'].split('_')[-1]
                    is_correct = row['event'].split('_')[-2]

                    if is_resp in ['noresp']:
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_resp'] = False
                    elif is_resp.isdigit():
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_resp'] = True
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'response_time'] = float(is_resp)
                    else:
                        raise ValueError(f"is_resp mismatch: {is_resp} in {row['event']}")

                    if is_correct in ['correct', 'incorrect']:
                        marker_df.loc[
                            marker_df['trial_index'] == row['trial_index'], 'is_correct'] = is_correct == 'correct'
                    else:
                        raise ValueError(f"is_correct mismatch: {is_correct} in {row['event']}")

            marker_df = marker_df.dropna(subset=['trial_index'])
            grouped = marker_df.groupby('trial_index')

            # Create a new DataFrame to hold the reformatted data
            formatted_data = pd.DataFrame()

            # Define the columns for the new DataFrame
            formatted_columns = ['trial_number', 'block_number', 'block_type', 'stim_indicator', 'subject_group',
                                 'stim', 'go_nogo', 'is_experienced', 'is_resp', 'is_correct', 'response_time',
                                 'response_time_real', 'trial_begin_time', 'stim_time',
                                 'trial_end_time', 'stim_desc', 'stim_type_info']

            # Initialize these columns in the formatted DataFrame
            for col in formatted_columns:
                formatted_data[col] = None

            # Iterate over each group, extract the times, and populate the new DataFrame
            for name, group in grouped:
                trial_index = int(name)
                trial_begin_time = group[group['event'].str.contains('trial-begin')]['time'].values[0]
                stim_time = group[group['event'].str.contains('stim')]['time'].values[0]
                trial_end_time = group[group['event'].str.contains('trial-end')]['time'].values[0]
                response_time = group['response_time'].values[0]

                """
                try:
                    response_time = group[group['event'].str.contains('response-received')]['time'].values[0]
                except:
                    response_time = None
                """

                stim_indicator = group['stim_indicator'].values[0]  # Assuming this value is constant within each trial
                go_nogo = group['go_nogo'].values[0]  # Assuming this value is constant within each trial
                is_resp = group['is_resp'].values[0]  # Assuming this value is constant within each trial
                is_correct = group['is_correct'].values[0]  # Assuming this value is constant within each trial
                block_type = group['block_type'].values[0]
                block_number = int(group['block_index'].values[0])
                stim = group['stim'].values[0]
                stim_desc = group['stim_desc'].values[0]
                is_experienced = stim == subject_group

                stim_row = group[group['event'].str.contains('stim_')].iloc[0] if not group[
                    group['event'].str.contains('stim_')].empty else None
                img_or_word_row = group[(group['event'].str.contains('img-')) | (group['event'].str.contains('word-'))]
                img_or_word_row = img_or_word_row['event'].values
                if len(img_or_word_row) == 0:
                    img_or_word_row = None

                # Add the row to the new DataFrame
                new_trial_row = {
                    'trial_number': trial_index,
                    'block_number': block_number,
                    'block_type': block_type,
                    'stim_indicator': stim_indicator,
                    'subject_group': subject_group,
                    'stim': stim,
                    'go_nogo': go_nogo,
                    'is_experienced': is_experienced,
                    'is_resp': is_resp,
                    'is_correct': is_correct,
                    'response_time': trial_end_time - stim_time,
                    'response_time_real': response_time,
                    'trial_begin_time': trial_begin_time,
                    'stim_time': stim_time,
                    'trial_end_time': trial_end_time,
                    'stim_desc': stim_row['event'],
                    'stim_type_info': stim_desc
                }
                for col in formatted_data.columns:
                    if formatted_data[col].dtype == 'object' and all(
                            isinstance(val, bool) for val in formatted_data[col].dropna()):
                        formatted_data[col] = formatted_data[col].astype(bool)
                new_trial_df = pd.DataFrame([new_trial_row.values()],
                                            columns=list(new_trial_row.keys()))
                formatted_data = pd.concat([formatted_data, new_trial_df], ignore_index=True)

            formatted_data.reset_index(drop=True, inplace=True)
            formatted_data['trial_number'] = pd.to_numeric(formatted_data['trial_number'],
                                                           errors='coerce')  # This will convert to numeric and set errors to NaN
            test_data_sorted = formatted_data.dropna(subset=['trial_number']).sort_values(
                by='trial_number')  # Drop rows where trial_index could not be converted
            test_data_sorted.set_index('trial_number', inplace=True)
            formatted_data.to_csv(self.data_directory + file_name.split('.')[0] + '_reformatted.csv', index=False)

            unique_blocks = formatted_data['block_number'].nunique()
            unique_block_types = [str(formatted_data[formatted_data['block_number'] == block_idx]['block_type'].unique()) for
                                  block_idx in formatted_data['block_number'].unique()]
            total_trials = formatted_data.shape[0]

            print(
                f"The marker contains {unique_blocks} blocks ({', '.join(unique_block_types)}) and {total_trials} trials")

        return formatted_data

    def _convert_continuous_to_trial(self, eeg_times, eeg_data, s_rate, formatted_marker_df, file_name,
                                     load_trialed_data=False,
                                     save_data=False):
        if load_trialed_data is True:
            with open(self.data_directory + file_name.split('.')[0] + '_trial_data.pkl', 'rb') as file:
                data_loaded = pickle.load(file)

            # Extract variables from the loaded dictionary
            eeg_data_array = data_loaded['eeg_data_array']
            eeg_time = data_loaded['eeg_time']
            eeg_labels = data_loaded['eeg_labels']
            trial_length = data_loaded['trial_length']
            trial_index = data_loaded['trial_index']
        else:
            trial_begin_times = formatted_marker_df['trial_begin_time']
            trial_onset_times = formatted_marker_df['stim_time']
            trial_response_times = formatted_marker_df['response_time']
            trial_end_times = formatted_marker_df['trial_end_time']

            trial_length = trial_end_times - trial_onset_times

            eeg_data_list, trial_index = [], []
            real_time, relative_time = [], []
            trial_length = []
            eeg_labels = {
                'block_number': [],
                'block_type': [],
                'stim_indicator': [],
                'go_nogo': [],
                'is_experienced': [],
                'is_resp': [],
                'is_correct': [],
                'stim': []
            }

            eeg_time = np.arange(int(- 2 * s_rate), int(s_rate)) / s_rate

            error_list = []
            for index, row in formatted_marker_df.iterrows():
                trial_begin_time = row['trial_begin_time']
                trial_onset_time = row['stim_time']
                trial_response_time = row['response_time']
                trial_end_time = row['trial_end_time']

                idx_start = np.argmin(np.abs(eeg_times - trial_begin_time))
                idx_end = np.argmin(np.abs(eeg_times - trial_end_time))
                idx_stim = np.argmin(np.abs(eeg_times - trial_onset_time))

                tl = (idx_end - idx_stim) / s_rate
                if tl < 0.8 * row['response_time']:
                    print(
                        f"Segmentation Error: The matched segment with Trial {row['trial_number']} has task duration "
                        f"{tl} which is less that expected duration {row['response_time']} ")
                    error_list.append(index)
                else:
                    eeg_labels['block_number'].append(row['block_number'])
                    eeg_labels['block_type'].append(row['block_type'])
                    eeg_labels['stim_indicator'].append(row['stim_indicator'])
                    eeg_labels['go_nogo'].append(row['go_nogo'])
                    eeg_labels['is_experienced'].append(row['is_experienced'])
                    eeg_labels['is_resp'].append(row['is_resp'])
                    eeg_labels['is_correct'].append(row['is_correct'])
                    eeg_labels['stim'].append(row['stim'])

                    trial_length.append((idx_end - idx_stim) / s_rate)

                    idx_start = int(idx_stim - 2 * s_rate)
                    idx_end = int(idx_stim + s_rate)

                    eeg_data_list.append(eeg_data[:, idx_start:idx_end])
                    real_time.append(eeg_times[idx_start:idx_end])

                    trial_index.append(row['trial_number'])

            eeg_data_array = np.stack(eeg_data_list)

            unique_blocks = np.unique(eeg_labels['block_number'])
            df = pd.DataFrame(eeg_labels)
            unique_block_types = [
                str(df[df['block_number'] == block_idx]['block_type'].unique()) for
                block_idx in df['block_number'].unique()]
            total_trials = eeg_data_array.shape[0]
            print(
                f"The trialed data contains {len(unique_blocks)} blocks ({', '.join(unique_block_types)}) and {total_trials} trials")

            if save_data is True:
                data_to_save = {
                    'eeg_data_array': eeg_data_array,
                    'eeg_time': eeg_time,
                    'eeg_labels': eeg_labels,
                    'trial_length': trial_length,
                    'trial_index': trial_index
                }

                # Save the dictionary into a file
                with open(self.data_directory + file_name.split('.')[0] + '_trial_data.pkl', 'wb') as file:
                    pickle.dump(data_to_save, file)

        return eeg_data_array, eeg_time, eeg_labels, trial_length, trial_index


class EEGDataSet:
    """
        A class to represent EEG dataset.

        Attributes
        ----------
        data : ndarray or None
            The EEG data.
        response_time : ndarray or None
            Array containing the response times of the trials.
        decision : ndarray or None
            Array containing the decision results of the trials.
        fs : float or None
            The sampling frequency of the EEG data.
        time_ms : ndarray or None
            Time in milliseconds for each data point.
        trial_block : ndarray or None
            The block number for each trial.
        trial_index : ndarray or None
            The index of each trial.
        trial_type : ndarray or None
            The type of each trial.
        channel_names : list of str or None
            Names of EEG channels.
        channel_index : ndarray or None
            Indices of the channels.
        channel_group : ndarray or None
            Group information of the channels.
        dgd_outputs : ndarray or None
            Outputs from the DGD process.



    """

    def __init__(self):
        """
        Initializes the EEGDataSet with default values.
        """
        self.data = None
        self.response_time = None
        self.decision = None
        self.labels = None
        self.fs = None
        self.time_ms = None
        self.trial_block = None
        self.trial_index = None
        self.trial_type = None
        self.channel_names = None
        self.channel_index = None
        self.channel_group = None
        self.bad_channels = None

        self.dgd_outputs = None

        self.file_name = None
        self.stream_id = None
        self.reference_electrodes = None

    def save_to_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
