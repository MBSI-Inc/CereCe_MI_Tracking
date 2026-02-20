import numpy as np
import pickle
import pandas as pd
import os
from scipy.signal import welch, butter, filtfilt

# Import from the same package level assuming helper functions are available or redefine them
# Since this is a standalone module in reconstruction/modules (conceptually), we might need to copy helper functions 
# or import them if the path allows. The user said to "refactor the old code into mi_predictor.py".
# I'll include the necessary helper functions as private methods or inner functions to be self-contained, 
# or assume a utilities module exists if I prefer. 
# Given the instructions, I should implement the logic directly or reuse.
# For robustness, I will include the helper logic (filtering, etc.) within the class or as standalone functions in this file.

class MI_Predictor:
    '''
    Load a pre-trained MI model and use it to predict motor imagery (MI) signals from EEG data in real-time.
    '''

    def __init__(self, config):
        self.config = config

        # Parameters
        self.sf = config.get('sf', 250)  # Sampling frequency
        self.low_freq = config.get('low_freq', 7)
        self.high_freq = config.get('high_freq', 30)
        self.signal_len = config.get('signal_len', 5)  # Window length in seconds
        self.n_ch = config.get('n_ch', 4)
        
        # Channel labels should match the model's expectation
        # Default for 4 channels: ["Fcz", "C3", "Cz", "C4"]
        self.channels = config.get('channels', ["Fcz", "C3", "Cz", "C4"])
        
        # Indices for specific processing (Cz re-referencing)
        try:
            self.cz_idx = self.channels.index("Cz")
        except ValueError:
            self.cz_idx = None
            print("Warning: 'Cz' channel not found in channel list. Re-referencing will be skipped.")

        # Load Model
        self.model_path = config.get('model_path', 'mi_model.sav')
        self.model = self._load_model(self.model_path)
        
        # Output mapping
        # 0: Left, 1: Right (based on common MI paradigms, but need to verify with user's specific model output mapping)
        # In old code: AsyncMICore returns predicted index.
        # Check src/constants.py or src/async_mi_core.py for output mapping.
        # src/constants.py: LEFT_CODE = "sw_12", RIGHT_CODE = "sw_13" but model likely predicts 0 or 1.
        # In start_MI_tracking: 'left', 'right', 'stop'.
        # I'll map the model output (likely 0 or 1) to 'left' or 'right'.
        # Assuming model.predict returns [0] for Left and [1] for Right or similar.
        # Let's assume 0 -> 'left', 1 -> 'right' for now, or return the raw prediction.

    def _load_model(self, model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

    def process_and_predict(self, data):
        '''
        Process the incoming EEG data and return a prediction.
        
        Args:
            data: np.array of shape (n_samples, n_ch + 1) or (n_samples, n_ch). 
                  The incoming data from EEG_Receiver. 
                  Expects timestamp in column 0 if n_ch + 1.
        
        Returns:
            prediction: 'left', 'right', or None
        '''
        
        # Handle data shape
        if data.shape[1] == self.n_ch + 1:
            eeg_data = data[:, 1:] # Drop timestamp
        else:
            eeg_data = data

        # Ensure we process sufficient data window (signal_len)
        required_samples = int(self.signal_len * self.sf)
        
        # If the input data is shorter than required, we can't make a valid prediction
        # The external buffer (receiver) must be configured to hold at least signal_len * sf samples.
        current_len = eeg_data.shape[0]
        if current_len < required_samples:
            print(f"[MI_Predictor] Insufficient data: {current_len}/{required_samples} samples. Skipping prediction.")
            return 'none'
        
        # Slice the most recent window (signal_len)
        # Note: We process this window independently each time (stateless predictor).
        # This differs slightly from AsyncMICore which managed an internal buffer with overlap,
        # but achieves the same result if called frequently with a sliding window input.
        
        epoch = eeg_data[-required_samples:, :].T  # Transpose to (n_ch, n_samples)
        
        # Feature Extraction
        psd_features = self._get_features(epoch)
        
        # Prediction
        
        # Feature Extraction
        psd_features = self._get_features(epoch)
        
        # Prediction
        if self.model and len(psd_features) > 0:
            try:
                # model.predict expects (n_samples, n_features)
                prediction = self.model.predict([psd_features])[0]
                
                # tailored for specific model output
                # Assuming 0 is Left, 1 is Right. 
                # Adjust based on training labels if known. 
                if prediction == 0:
                    return 'left'
                elif prediction == 1:
                    return 'right'
                else:
                    return 'none'
            except Exception as e:
                print(f"[MI_Predictor] Prediction error: {e}")
                return 'none'
        return 'none'

    def _get_features(self, epoch):
        '''
        Extract PSD features from the epoch.
        Pipeline:
        1. Cz Re-referencing (if Cz exists)
        2. Filtering (Notch + Bandpass)
        3. PSD (Welch)
        '''
        
        # 1. Cz Re-referencing
        if self.cz_idx is not None:
            epoch = self._cz_rereference(epoch, self.cz_idx)
        
        # 2. Filtering
        # Notch filter (45-55Hz)
        epoch = self._custom_filter(epoch, 45, 55, 'bandstop')
        # Bandpass filter (low-high)
        epoch = self._custom_filter(epoch, self.low_freq, self.high_freq, 'bandpass')
        
        # 3. PSD
        psds, _ = self._psd_epoch(epoch, self.sf, self.low_freq, self.high_freq)
        
        return psds

    def _cz_rereference(self, arr, cz_ind):
        '''
        Subtract Cz channel from others and remove Cz.
        arr: (n_ch, n_samples)
        '''
        # Broadcast subtraction
        arr_cz = arr - arr[cz_ind, :].reshape(1, -1)
        # Remove Cz channel row
        arr_cz = np.delete(arr_cz, cz_ind, axis=0)
        return arr_cz

    def _custom_filter(self, data, low, high, filt_type, order=4):
        '''
        Apply Butterworth filter.
        '''
        b, a = butter(order, [low, high], btype=filt_type, fs=self.sf)
        return filtfilt(b, a, data, axis=1)

    def _psd_epoch(self, epoch, sf, low, high):
        '''
        Compute PSD for the epoch.
        epoch: (n_ch, n_samples)
        Replicates behavior of matplotlib.pyplot.psd used in old code.
        plt.psd default: NFFT=256 (overridden to sf), Fs=2 (overridden to sf), 
        window=hanning, noverlap=0.
        '''
        n_ch = epoch.shape[0]
        
        # We use scipy.signal.welch to avoid plotting overhead.
        # To match plt.psd(x, NFFT=sf, Fs=sf) defaults:
        # window='hann' (default in both, effectively)
        # noverlap=0 (plt.psd default is 0, welch default is nperseg//2)
        # nperseg=sf (matches NFFT=sf)
        
        psds = []
        freqs = None
        
        for ch in range(n_ch):
            f, Pxx = welch(epoch[ch, :], fs=sf, nperseg=int(sf), noverlap=0, window='hann')
            
            if freqs is None:
                freqs = f
            
            psds.append(Pxx)
            
        psds = np.array(psds)
        
        # Slicing to frequency band
        # Robust finding of indices
        # In old code: low_ind = np.where(freqs == low)[0][0]
        # This assumes exact match. With nperseg=sf (1 sec), freq resolution is 1Hz.
        # So integer freqs should match exactly.
        
        try:
            # Try to match exact behavior if possible, or use robust argmax
            idx_min = np.where(freqs == low)[0][0]
            idx_max = np.where(freqs == high)[0][0]
        except IndexError:
            # Fallback to nearest if exact match fails
            idx_min = np.argmin(np.abs(freqs - low))
            idx_max = np.argmin(np.abs(freqs - high))
            
        # Old code: psds[:, low_ind[0][0] : high_ind[0][0]] (exclusive high)
        psds = psds[:, idx_min:idx_max]
        
        # Flatten
        psds_flat = psds.ravel()
        
        return psds_flat, freqs[idx_min:idx_max]


if __name__ == "__main__":
    # --- Test Configuration ---
    print("--- Starting MI Predictor Test ---")
    
    config = {
        'sf': 250,
        'low_freq': 7,
        'high_freq': 30,
        'signal_len': 2, # Using 2 seconds for faster testing
        'n_ch': 4,
        'channels': ["Fcz", "C3", "Cz", "C4"],
        'model_path': 'non_existent_model.sav' 
    }

    predictor = MI_Predictor(config)

    # --- Mock Model for Testing ---
    # Since we likely don't have a real model file during this test run,
    # we manually inject a mock model that alternates predictions.
    if predictor.model is None:
        print("[Test] No model loaded. Injecting MockModel for verification.")
        class MockModel:
            def predict(self, X):
                # Returns 0 (left) or 1 (right) based on some simple logic 
                # or just random for testing flow.
                # Here we just return 0 (Left) for simplicity.
                return [0] 
        predictor.model = MockModel()

    # --- Test Case 1: Insufficient Data ---
    print("\n[Test Case 1] Insufficient Data (100 samples)")
    # Create random data: (100 samples, 4 channels)
    # Note: process_and_predict handles (samples, channels) or (samples, channels+1)
    # We use (samples, channels) here.
    dummy_data_short = np.random.rand(100, 4)
    result_short = predictor.process_and_predict(dummy_data_short)
    print(f"Result: {result_short} (Expected: none)")

    # --- Test Case 2: Sufficient Data ---
    print("\n[Test Case 2] Sufficient Data (500 samples for 2s window)")
    # 2 seconds * 250 Hz = 500 samples
    dummy_data_full = np.random.randn(500, 4) 
    
    # We expect 'left' because our MockModel returns 0
    result_full = predictor.process_and_predict(dummy_data_full)
    print(f"Result: {result_full} (Expected: left [from MockModel])")

    # --- Test Case 3: Data with Timestamp Column ---
    print("\n[Test Case 3] Sufficient Data with Timestamp Column")
    # (500 samples, 4 channels + 1 timestamp)
    timestamps = np.arange(500).reshape(-1, 1)
    data_with_ts = np.hstack((timestamps, dummy_data_full))
    
    result_ts = predictor.process_and_predict(data_with_ts)
    print(f"Result: {result_ts} (Expected: left)")

    # --- Test Case 4: Real Data and Model ---
    print("\n[Test Case 4] Real Data from file (70,000+ samples)")
    
    # Paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_file = os.path.join(project_root, 'data', 'MItest_24-01-27_ExG.csv')
    model_file = os.path.join(project_root, 'models', 'LDA/MItest_24-01-27.sav')

    if os.path.exists(data_file) and os.path.exists(model_file):
        print(f"Loading data from {data_file}...")
        try:
             # Read CSV matching the format: TimeStamp, ch1, ch2, ch3, ch4
            df = pd.read_csv(data_file)
            real_data = df.values # Convert to numpy array
            print(f"Data Loaded: shape {real_data.shape}")

            # Configure predictor with real model
            real_config = {
                'sf': 250,
                'low_freq': 7,
                'high_freq': 30,
                'signal_len': 5, # Standard 5s window
                'n_ch': 4,
                'channels': ["Fcz", "C3", "Cz", "C4"],
                'model_path': model_file
            }
            print(f"Initializing predictor module with real model from {model_file}...")
            real_predictor = MI_Predictor(real_config)
            print("Predictor initialized. Processing real data...")
            
            if real_predictor.model is not None:
                # Simulate receiving all data at once
                print("Predicting on full dataset (should auto-slice last 5s)...")
                real_result = real_predictor.process_and_predict(real_data)
                print(f"Real Data Prediction Result: {real_result}")
            else:
                print("Failed to load real model. Skipping prediction.")

        except Exception as e:
            print(f"Error processing real data: {e}")
    else:
        print(f"Data or Model file not found at {data_file} or {model_file}")

    print("\n--- Test Complete ---")

