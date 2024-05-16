import os
import pickle

import numpy as np
from scipy import signal
from scipy.signal import detrend


class DataPreprocessor:
    """
    A class for preprocessing EEG data.

    This class includes methods for various preprocessing steps such as
    baseline removal, resampling, common average referencing, filtering,
    and removing line noise.

    Attributes
    ----------
    time : numpy array or None
        Time vector for EEG data.
    fs : float or None
        Sampling frequency of the EEG data.
    """

    def __init__(self, paths, settings):
        """
        Initializes the DataPreprocessor with default values.
        """
        self.time = None
        self.fs = None
        self.paths = paths
        self.settings = settings

    def preprocess(self, eeg_dataset):
        """
        Preprocess all patient data in the EEGDataSet.

        This method iterates over all patient data in the provided EEGDataSet instance
        and applies preprocessing to the EEG data.

        Parameters
        ----------
        eeg_dataset : EEGDataSet
            The EEGDataSet instance containing patient data.

        Returns
        -------
        EEGDataSet
            The EEGDataSet instance with preprocessed data.
        """
        "=================== Preprocess Data =================="
        for patient_id, data in eeg_dataset.all_patient_data.items():


            patient_dataset = eeg_dataset.all_patient_data[patient_id]
            self.time = patient_dataset.time_ms
            self.fs = patient_dataset.fs

            print(
                f"Subject {patient_id} from {len(eeg_dataset.all_patient_data.items())}: {patient_dataset.file_name.split('.')[0]} Preprocess Data")

            preprocessed_data_path = self.paths.xdf_directory_path + patient_dataset.file_name.split('.')[0] + "_preprocessed.pkl"
            if os.path.exists(preprocessed_data_path) and self.settings.load_preprocessed_data is True:
                with open(preprocessed_data_path, 'rb') as file:
                    preprocessed_dataset = pickle.load(file)
            else:
                preprocessed_dataset = self.apply_preprocessing(patient_dataset.data)
                if self.settings.save_preprocessed_data is True:
                    with open(preprocessed_data_path, 'wb') as file:
                        pickle.dump(preprocessed_dataset, file)

            eeg_dataset.all_patient_data[patient_id].data = preprocessed_dataset

        return eeg_dataset

    def apply_preprocessing(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a series of preprocessing steps to the EEG data.

        This method can include baseline removal and other preprocessing steps
        defined in separate methods.

        Parameters
        ----------
        data : numpy.ndarray
            The EEG data to be preprocessed.

        Returns
        -------
        numpy.ndarray
            Preprocessed EEG data.
        """
        data = self.remove_baseline(data, normalize=False, baseline_t_min=-1000)
        # data = self.detrend_eeg(data)
        # data = self.common_average_referencing(data)
        # Additional preprocessing steps can be added here
        return data

    def remove_baseline(self, data, baseline_t_min=-300, baseline_t_max=0, normalize=True):
        """
        Remove the baseline from the EEG data using specified time window.

        This method calculates and subtracts the mean signal in the specified
        baseline time window for each trial and channel.

        Parameters
        ----------
        data : numpy.ndarray
            The EEG data from which to remove the baseline.
        baseline_t_min : int, optional
            The starting time (in ms) of the baseline period (default is -300).
        baseline_t_max : int, optional
            The ending time (in ms) of the baseline period (default is 0).

        Returns
        -------
        numpy.ndarray
            The EEG data with the baseline removed.
        """
        # Determine start and end indices for the baseline time window
        idx_start = np.argmin(np.abs(self.time - baseline_t_min))
        idx_end = np.argmin(np.abs(self.time - baseline_t_max))

        if (idx_end - idx_start) > 5:
            # Calculate the baseline mean values for each trial and channel
            baseline_means = np.mean(data[:, :, idx_start:idx_end], axis=2, keepdims=True)

            # Subtract the baseline mean from all time points for each trial and channel
            eeg_data_baseline_removed = data - baseline_means

            if normalize is True:
                maximum = np.quantile(np.abs(eeg_data_baseline_removed[:, :, idx_start:idx_end]),
                                      0.99, axis=2, keepdims=True)
                eeg_data_baseline_removed = eeg_data_baseline_removed / maximum

        return eeg_data_baseline_removed

    def detrend_eeg(self, data):
        """
        Detrend the EEG data by removing the linear trend from each trial and channel.

        This method applies a linear detrend operation, which can be useful for
        reducing low-frequency drifts in the EEG data.

        Parameters
        ----------
        data : numpy.ndarray
            The EEG data from which to remove the linear trend.

        Returns
        -------
        numpy.ndarray
            The EEG data with the linear trend removed.
        """
        # Apply linear detrend on each trial and channel
        eeg_data_detrended = np.apply_along_axis(detrend, axis=2, arr=data)

        return eeg_data_detrended

    def resample(self, f_resample, anti_alias_filter=False):
        """
        Resample the EEG data to a specified frequency.

        If 'anti_alias_filter' is True, the data is filtered before resampling
        to prevent aliasing.

        Parameters
        ----------
        f_resample : float
            The new sampling frequency.
        anti_alias_filter : bool, optional
            Whether to apply an anti-aliasing filter before resampling (default is False).

        Returns
        -------
        tuple
            Tuple containing the updated time vector and resampled data.
        """
        # chane data into mne epochs
        epochs = self.array_to_mne_epoch()

        decim_factor = int(self.fs / f_resample)  # Replace this with your desired decimation factor

        if anti_alias_filter is True:
            # re-sample first filter data to prevent aliasing
            epochs_down_sampled = epochs.copy().resample(f_resample)
        else:
            epochs_down_sampled = epochs.copy().decimate(decim_factor)

        self.data = epochs_down_sampled.get_data()

        self.time = self.time[0] + epochs_down_sampled.times
        self.fs = epochs_down_sampled.info['sfreq']
        self.dataset.fs = epochs_down_sampled.info['sfreq']
        self.dataset.data = self.data
        self.dataset.time = self.time

        return self.time, self.data

    def common_average_referencing(self, data):
        """
        Apply common average referencing to the EEG data.

        This method computes the mean signal across all electrodes at each time point
        and subtracts it from each electrode's signal.

        Returns
        -------
        numpy.ndarray
            The EEG data after applying common average referencing.
        """
        # Compute the mean signal across all electrodes for each time point
        mean_signal = np.mean(data, axis=1, keepdims=True)
        # Subtract the mean signal from each electrode's signal
        data = data - mean_signal

        return data

    def filter_data(self, low_cutoff=0.1, high_cutoff=300):
        """
        Filter the EEG data using a bandpass filter.

        This method applies a Butterworth bandpass filter to the data.

        Parameters
        ----------
        low_cutoff : float, optional
            The low-frequency cutoff for the filter (default is 0.1 Hz).
        high_cutoff : float, optional
            The high-frequency cutoff for the filter (default is 300 Hz).

        Returns
        -------
        numpy.ndarray
            The filtered EEG data.
        """
        # Compute filter coefficients
        nyquist_freq = self.fs / 2
        b, a = signal.butter(4, [low_cutoff / nyquist_freq, high_cutoff / nyquist_freq], btype='bandpass')

        # Filter iEEG data
        filtered_data = signal.filtfilt(b, a, self.data, axis=2)

    def remove_line_noise(self,
                          freqs_to_remove=[60],
                          filter_length='auto',
                          phase='zero',
                          method='fir',
                          trans_bandwidth=1.0,
                          mt_bandwidth=None):
        # Reshape data to 2D array of shape (n_channels, n_samples)
        data_mne = self.array_to_mne_raw()

        # Apply the notch filter
        notch_filtered_epochs = data_mne.notch_filter(freqs_to_remove,
                                                      filter_length=filter_length,
                                                      phase=phase,
                                                      method=method,
                                                      trans_bandwidth=trans_bandwidth,
                                                      mt_bandwidth=mt_bandwidth)

        filtered_data = notch_filtered_epochs.get_data()

        self.data = filtered_data.reshape(self.n_channels, self.n_trials, self.n_samples).transpose(1, 0, 2)
        self.dataset.data = self.data
        return self.data

    def baseline_removal(self, method='zero_mean', window_size=100, min_time=0, max_time=None):
        """
        Remove baseline drifts from data using one of several methods.

        Parameters
        ----------
        method : str, optional
            The baseline removal method to use. Defaults to 'zero_mean'.
            Options are: 'zero_mean', 'mean_norm', 'poly_fit', 'mov_avg', 'median'.
        window_size : int, optional
            The size of the window to use for the moving average or median filter methods.
            Defaults to 100.
        min_time : int, optional
            The minimum time index of the pre-stimulus period. Defaults to 0.
        max_time : int or None, optional
            The maximum time index of the pre-stimulus period. Defaults to None.

        Returns
        -------
        data_out : numpy array
            The baseline-removed data, of shape (n_epochs, n_channels, n_samples).
        """
        data = self.data

        if min_time is not None:
            min_time_idx = np.min(np.where(self.time > min_time))
        else:
            min_time_idx = 0

        if max_time is not None:
            max_time_idx = np.max(np.where(self.time < max_time))
        else:
            max_time_idx = len(self.time)

        # Calculate mean and standard deviation of the data for the pre-stimulus period
        data_mean = np.mean(data[:, :, min_time_idx:max_time_idx], axis=-1, keepdims=True)
        data_std = np.std(data[:, :, min_time_idx:max_time_idx], axis=-1, keepdims=True)

        # Apply baseline removal method
        if method == 'zero_mean':
            data_out = data - data_mean
        elif method == 'mean_norm':
            data_out = data / data_mean
        elif method == 'poly_fit':
            data_out = signal.detrend(data, axis=-1, type='linear')
        elif method == 'mov_avg':
            window = signal.windows.hann(window_size)
            data_out = signal.convolve(data, window[None, None, :], mode='same', axis=-1) / np.sum(window)
            data_out -= data_mean
        elif method == 'median':
            data_out = signal.medfilt(data, kernel_size=(1, 1, window_size))
            data_out -= data_mean
        else:
            raise ValueError("Invalid baseline removal method.")

        # Normalize data by standard deviation
        data_out /= data_std
        self.data = data_out
        return self.data
