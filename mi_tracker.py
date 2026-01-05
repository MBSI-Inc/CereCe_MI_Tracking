# One key idea is to keep the EEG sampling (250 Hz), PSD computation/MI prediction updates, and wheelchair commands (left/right/stop) in sync.
# we can settle this by adjusting window sizes, step sizes to ensure they are synchronized.
class MI_Tracker:
    '''
    Load a pre-trained MI model and use it to predict motor imagery (MI) signals from EEG data in real-time.

    pipeline includes:
    - connect to an EEG device (e.g., OpenBCI, Muse, etc.)
    - preprocess EEG data (e.g., bandpass filtering, artifact removal, etc.)
    - compute power spectral density (PSD) features from EEG data
    - use the pre-trained MI model to predict MI signals (left/right/no MI)
    - return the predicted MI signal for wheelchair control

    config dict should include:
    - 'eeg_device': name of EEG device to connect to (e.g., 'OpenBCI', 'Muse', etc.)
    - 'channels': list of EEG channels to use for MI prediction
    - 'eeg_sampling_rate': Sampling rate of the EEG device (e.g., 250 Hz)

    - 'low freq': Low cutoff frequency for bandpass filter (e.g., 8 Hz)
    - 'high_freq': High cutoff frequency for bandpass filter (e.g., 30 Hz)
    - 'psd_window_size': Window size for PSD computation (e.g., 1 second)
    - 'psd_step_size': Step size for PSD computation (e.g., 0.2 seconds)

    - 'mi_model_path': Path to the pre-trained MI model file
    
    - 'mock_file': (optional) path to a mock EEG data file for testing without a real EEG device
    
    '''

    def __init__(self, config):
        self.config = config

        self.eeg_device = config.get('eeg_device', 'OpenBCI')
        self.eeg_sampling_rate = config.get('eeg_sampling_rate', 250)
        self.channels = config.get('channels', [])

        self.low_freq = config.get('low_freq', 8)
        self.high_freq = config.get('high_freq', 30)
        self.psd_window_size = config.get('psd_window_size', 1) # in seconds
        self.psd_step_size = config.get('psd_step_size', 0.2) # in seconds 

        self.mi_model_path = config.get('mi_model_path', 'mi_model.pkl')

        self.mock_file = config.get('mock_file', None)

        # the following attributes will be updated sychronously when new EEG data is received and processed
        self.eeg_segmented_raw_data = None  # to store segmented raw EEG data. [channels x samples]
        self.mi_signal = -1  # -1 indicates no MI detected, 0 for left MI, 1 for right MI
        

    def eeg_device_connect(self):
        '''
        Connect to the specified EEG device based on config.

        '''
        eeg_device_name = self.config.get('eeg_device', 'OpenBCI')
        # Implement connection logic for different EEG devices here
        pass  # Replace with actual implementation

    def eeg_device_disconnect(self):
        '''
        Disconnect from the EEG device.

        '''
        pass  # Replace with actual implementation


    def eeg_device_get_data(self):
        '''
        Get raw EEG data from the connected EEG device, and fill it into a data buffer.

        Returns:
            eeg_data: numpy array of shape (n_channels, n_samples)
        '''
        pass  # Replace with actual implementation


    def update_MI_signal(self, eeg_data):
        '''
        Update the MI signal based on new EEG data.

        eeg_data: new EEG data segment for processing
        '''
        # Preprocess EEG data (e.g., bandpass filtering)
        # Compute PSD features
        # Use pre-trained MI model to predict MI signal
        # Update self.mi_signal based on prediction

        pass  # Replace with actual implementation


    def get_MI_signal(self):
        return self.mi_signal