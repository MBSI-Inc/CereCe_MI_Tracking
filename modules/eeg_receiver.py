# Load in helper functions
from explorepy import Explore
from .mockExplore import MockExplore
from .helper_functions import *
from .self_threaded_process_interface import AbstractSelfThreadedProcess

# Online motor imagery experiment
import pickle
import time
import os
import numpy as np
from . import constants
from threading import Lock, Thread
from explorepy.stream_processor import TOPICS

class EEG_Receiver(AbstractSelfThreadedProcess):
    """
    This is a thread that receives EEG data from the Explore device.
    It maintains a buffer of the most recent EEG data for processing.

    It allows EEG signal from device or from a mock CSV file for testing.

    Args:
        eeg_name (str): Name of the Explore device to connect to.
        filename (str): Path to the pre-trained model file for classification.
        mock_file (str, optional): Path to a CSV file for mock data. Defaults to None.
        silent (bool, optional): If True, suppresses console output. Defaults to True.
    Returns:
        EEG_Receiver: An instance of the EEG_Receiver class.
    """

    def __init__(self, eeg_name, use_mock=False, mock_file=None, silent=True):

        # mock data parameters
        self.use_mock = use_mock
        self.mock_file = mock_file

        # EEG device parameters
        self.eeg_name = eeg_name
        self.CH_LABELS = ["Fcz", "C3", "Cz", "C4"]
        self.C3_ch = self.CH_LABELS.index("C3")
        self.C4_ch = self.CH_LABELS.index("C4")
        self.Cz_ch = self.CH_LABELS.index("Cz")
        self.Fcz_ch = self.CH_LABELS.index("Fcz")
        self.n_ch = 4

        self.explore = self.load_device_or_mock()
        
        
        # signal processing parameters
        self.sf = constants.SAMPLING_FREQ
        self.signal_len = constants.SIGNAL_LENGTH
        
        # Buffer parameters
        self._data_buff = np.array([])

        # threading parameters
        self.thread = None
        self.lock = Lock()
        self.running = False
        self.silent = silent

        
    
    def load_device_or_mock(self):
        '''
        Load Explore device or MockExplore based on configuration.
        args:
            self.use_mock: bool, whether to use mock data
            self.mock_file: str, path to mock data file
            self.eeg_name: str, name of the EEG device
        returns:
            explore: Explore or MockExplore object
        '''
        if self.use_mock and self.mock_file is not None:
            if not os.path.isfile(self.mock_file):
                raise FileNotFoundError(
                    "Mock file not found: {}".format(self.mock_file)
                )
            else:
                print("Using mock data from file: {}".format(self.mock_file))
                explore = MockExplore(
                    csv_path=self.mock_file,
                    chunk_size=constants.CHUNK_SIZE,  # how many samples per chunk
                    sampling_rate=constants.SAMPLING_FREQ,  # the sampling frequency of your original data
                )
        else:
            print("Connecting to Explore device: {}".format(self.eeg_name))
            explore = Explore()
        
        return explore





    def __run(self):
        # Fill buffer
        if not self.silent:
            print("Waiting for initial buffer load...")
        time.sleep(5)

        while True:
            self.__analyse_data()  # outputs self.predicted

            if not self.silent:
                print(
                    "## Async MI: prediction: {0}, raw prediction: {1}".format(
                        self.predicted, self.predicted_raw
                    )
                )

            if not self.running:
                break

            # pause processing for the time expected to take to re-fill the buffer
            if not self.silent:
                print(
                    "run loop sleeping for {0} seconds".format(
                        (self.signal_len - self.overlap) / 2
                    )
                )
            time.sleep((self.signal_len - self.overlap) / 2)

    # public
    def start(self):
        if not self.running:
            self.running = True

            # Connect to EEG
            self.explore.connect(device_name=self.eeg_name)

            self.explore.stream_processor.subscribe(
                callback=self.update_buffer, topic=TOPICS.raw_ExG
            )

            # Setting thread as daemon ensure this is killed even if stop is not called.
            # EG this class is disposed but stop is not called.
            # Setting may not be necessary, but probably a good precaution.
            self.thread = Thread(target=self.__run, daemon=True)
            self.thread.start()


    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()
            self.explore.disconnect()


    def update_buffer(self, packet):
        """Update EEG buffer of the experiment
        Args:
            packet (explorepy.packet.EEG): EEG packet
        """
        _, eeg = packet.get_data()
        if not len(self._data_buff):  # if data_buff array is empty
            self._data_buff = eeg.T  # load eeg data into array
        else:
            self._data_buff = np.concatenate(
                (self._data_buff, eeg.T), axis=0
            )  # add to the end of the existing array

    def __analyse_data(self):
        # get data from update_buffer
        # analyse it using classify
        # return predicted_hand

        if len(self._data_buff) > 0:  # if there is data loaded into data_buff array
            if (
                self._data_buff.shape[0] > self.signal_len * self.sf
            ):  # if data_buff gets to the length of sliding window
                # When the lock exists, all other threads cannot access the lock.
                # The purpose of this lock is to prevent any other threads from modifying the data buffer in the middle of analysis.
                with self.lock:
                    self.epoch = self._data_buff[
                        : int(self.signal_len * self.sf), :
                    ].T  # slice the end of the array and transpose it
                    self.predicted = self.__classify(self.epoch)  # classify it
                    self.epoch = np.array([])  # Just incase

                    self.predicted_raw = self.predicted
                    self.predicted = self.__exponential_filter(
                        value=self.predicted, value_previous=self.predicted_previous
                    )  # Smooth it

                    self.predicted_previous = self.predicted

                    # Desired outcome is for the end section of the _data_buffer of size equal to overlap to be kept each time.
                    # Doing it in specifically this way prevents this buffer from going faster than we can crop it.
                    overlap_sample_count = int(self.overlap * self.sf)
                    samples_to_chop = len(self._data_buff) - overlap_sample_count
                    self._data_buff = self._data_buff[
                        samples_to_chop:, :
                    ]  # sliding window

    # public
    def getData(self):
        return (self.predicted, self.predicted_raw)

    def __del__(self):
        self.stop()

    def __exponential_filter(self, value, value_previous):
        return ((value_previous + 1) * self.beta) + ((value + 1) * (1 - self.beta)) - 1

    def __classify(self, epoch):
        psd = self.__get_features(epoch)
        result = self.loaded_model.predict([psd])
        return result[0]

    def __get_features(self, epoch):
        # Cz re-reference
        Cz_ind = self.CH_LABELS.index("Cz")
        epoch_Cz = Cz_rereference(epoch, Cz_ind)
        # Filter
        epoch_filt = filter_epoch(
            epoch=epoch_Cz, low=self.low, high=self.high, sf=self.sf
        )
        epoch_psd, _ = PSD_epoch(
            epoch=epoch_filt, low=self.low, high=self.high, sf=self.sf
        )
        return epoch_psd