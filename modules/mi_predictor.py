import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


class MI_Predictor:
    '''
    Classifies EEG as MI-active or rest using the Cerebruh pipeline.
    Matches the feature extraction in Cerebruh_BCI/Experiments/MI_2/src exactly:
    Cz re-reference → notch + bandpass → plt.psd(NFFT=sf, Fs=sf) → 7–29 Hz bins (exclusive high).
    Returns 'active' (model output 1) or 'none' (model output 0).
    '''

    def __init__(self, config):
        self.sf          = config.get('sf', 250)
        self.low_freq    = config.get('low_freq', 7)
        self.high_freq   = config.get('high_freq', 30)
        self.signal_len  = config.get('signal_len', 5)
        self.n_ch        = config.get('n_ch', 4)
        self.channels    = config.get('channels', ["Fcz", "C3", "Cz", "C4"])

        try:
            self.cz_idx = self.channels.index("Cz")
        except ValueError:
            self.cz_idx = None
            print("Warning: 'Cz' not in channel list — re-referencing skipped.")

        self.model_path = config.get('model_path', 'models/LDA/MItest_24-01-27.sav')
        self.model = self._load_model(self.model_path)

    def _load_model(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            print(f"[MI_Predictor] Model load failed — missing module: {e}")
            return None
        except Exception as e:
            print(f"[MI_Predictor] Model load failed: {e}")
            return None

    def process_and_predict(self, data):
        '''
        Args:
            data: np.array shape (n_samples, n_ch+1) with timestamp col 0,
                  or (n_samples, n_ch) without.
        Returns:
            'active' if MI detected, 'none' otherwise.
        '''
        if data.shape[1] == self.n_ch + 1:
            eeg_data = data[:, 1:]
        else:
            eeg_data = data

        required_samples = int(self.signal_len * self.sf)
        if eeg_data.shape[0] < required_samples:
            return 'none'

        # Take the most recent window, transpose to (n_ch, n_samples)
        epoch = eeg_data[-required_samples:, :].T

        psd_features = self._get_features(epoch)
        if self.model is None or len(psd_features) == 0:
            return 'none'

        try:
            prediction = self.model.predict([psd_features])[0]
            return 'active' if prediction == 1 else 'none'
        except Exception as e:
            print(f"[MI_Predictor] Prediction error: {e}")
            return 'none'

    def _get_features(self, epoch):
        # 1. Cz re-reference
        if self.cz_idx is not None:
            epoch = self._cz_rereference(epoch, self.cz_idx)
        # 2. Notch then bandpass
        epoch = self._custom_filter(epoch, 45, 55, 'bandstop')
        epoch = self._custom_filter(epoch, self.low_freq, self.high_freq, 'bandpass')
        # 3. PSD
        psds, _ = self._psd_epoch(epoch, self.sf, self.low_freq, self.high_freq)
        return psds

    def _cz_rereference(self, arr, cz_ind):
        arr_cz = arr - arr[cz_ind, :].reshape(1, -1)
        return np.delete(arr_cz, cz_ind, axis=0)

    def _custom_filter(self, data, low, high, filt_type, order=4):
        b, a = butter(order, [low, high], btype=filt_type, fs=self.sf)
        return filtfilt(b, a, data, axis=1)

    def _psd_epoch(self, epoch, sf, low, high):
        '''
        Matches plt.psd(x, NFFT=sf, Fs=sf) from Cerebruh helper_functions.py exactly.
        Frequency bins are integers with 1 Hz resolution; slicing is exclusive of high
        (same as original: psds[:, low_ind:high_ind]).
        '''
        n_ch = epoch.shape[0]
        psds = []
        freqs = None

        for ch in range(n_ch):
            psd, f = plt.psd(epoch[ch, :], NFFT=sf, Fs=sf)
            plt.close()
            if freqs is None:
                freqs = f
            psds.append(psd)

        psds = np.array(psds)

        low_ind  = np.where(freqs == low)[0][0]
        high_ind = np.where(freqs == high)[0][0]
        psds  = psds[:, low_ind:high_ind]   # exclusive high — matches training
        freqs = freqs[low_ind:high_ind]

        return psds.ravel(), freqs


if __name__ == "__main__":
    import pandas as pd

    print("--- MI Predictor self-test ---")

    config = {
        'sf': 250, 'low_freq': 7, 'high_freq': 30,
        'signal_len': 5, 'n_ch': 4,
        'channels': ["Fcz", "C3", "Cz", "C4"],
        'model_path': 'models/LDA/MItest_24-01-27.sav',
    }
    predictor = MI_Predictor(config)

    # Verify feature length: 3 post-Cz channels × 23 bins (7–29 Hz) = 69
    dummy = np.random.randn(int(config['signal_len'] * config['sf']), config['n_ch'])
    features = predictor._get_features(dummy.T)
    print(f"Feature length: {len(features)} (expected 69)")

    print("\n[Test 1] Insufficient data →", predictor.process_and_predict(np.random.randn(100, 4)))

    print("\n[Test 2] Sufficient data →", predictor.process_and_predict(np.random.randn(1250, 4)))

    data_file  = 'data/MItest_24-01-27_ExG.csv'
    model_file = 'models/LDA/MItest_24-01-27.sav'
    if os.path.exists(data_file) and os.path.exists(model_file):
        print("\n[Test 3] Real data →", end=' ')
        df = pd.read_csv(data_file)
        print(predictor.process_and_predict(df.values))
    else:
        print("\n[Test 3] Skipped — data/model not found")

    print("\n--- Done ---")
