from threading import Thread, Lock
from collections import deque
import numpy as np
import pandas as pd
import time
import os
import sys
import logging

# explorepy is imported lazily in device mode: importing it at module level
# pulls in its dashboard/sentry side-effects (notably on Windows) and breaks
# file-mode replay on machines without the SDK installed.
TOPICS = None

_log = logging.getLogger(__name__)


class EEG_Receiver(Thread):
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

    def __init__(self, params):
        super().__init__()
        self.mode = params.get('input_mode', 'device')    # input from 'file' or 'device'
        self.running = False
        self.daemon = True
        self.buffer = deque(maxlen=params.get('buffer_size', 300)) # buffer shape: (buffer_size(timestamp), n_ch)
        self.buffer_lock = Lock()
        self.connected = False
        self.last_data_time = None
        # Set once the source is exhausted (file mode) or acquisition ends,
        # so callers can tell "no more data is coming" from "data is late".
        self.finished = False


        if self.mode == 'file':
            print(f"[Receiver] Input Mode: File")
            self.data_path = params.get('data_path', None)
            self.explorer = self.simulate_device()
            self.connected = True

        elif self.mode == 'device':
            print(f"[Receiver] Input Mode: Device")
            self.device_name = params.get('device_name', 'Explore_842F')

            try:
                global TOPICS
                from explorepy import Explore
                from explorepy.stream_processor import TOPICS
            except Exception as e:
                # No SDK on this machine: replay from CSV if one is configured.
                self.data_path = params.get('data_path', None)
                if self.data_path:
                    print(f"[Receiver] Warning: explorepy import failed ({e}); "
                          "switching to file mode using data_path.")
                    self.mode = 'file'
                    self.explorer = self.simulate_device()
                    self.connected = True
                    return
                raise ImportError(
                    f"Could not import explorepy: {e}. Install explorepy or set "
                    "'input_mode' to 'file' with a 'data_path'."
                )

            if sys.platform.startswith('linux'):
                # The Mentalab C++ SDK drops the RFCOMM connection ~45ms after
                # streaming starts on Linux. Replace it with a pure Python socket
                # shim that does exact reads and stays stable.
                from explorepy import btcpp as _btcpp
                from modules import explorepy_linux_socket_shim as _shim
                _btcpp.exploresdk = _shim

                # The C++ error-report prompt blocks stdin; redirect to /dev/null
                # so it auto-answers with EOF.
                _null_fd = os.open('/dev/null', os.O_RDONLY)
                os.dup2(_null_fd, 0)
                os.close(_null_fd)

            self.explorer = Explore()
            self.explorer.connect(device_name=self.device_name)
            self.connected = True
        else:
            raise ValueError(f"Unknown mode: {self.mode}")



    def run(self):
        self.running = True
        if self.mode == 'device':
            print("[Receiver] Thread started. listening to hardware...")
            self.explorer.stream_processor.subscribe(
                    callback=self.update_buffer,
                    topic=TOPICS.raw_ExG
                )
            # The parser thread started by connect() streams data to our callback;
            # we just keep this thread alive until stop() is called.
            while self.running:
                time.sleep(0.1)

        elif self.mode == 'file':
            print(f"[Receiver] Thread started. reading from file: {self.data_path}...")
            if self.explorer:
                 self.explorer.acquire()
            self.finished = True
            print("[Receiver] Source exhausted (file mode).")


    def stop(self):
        self.running = False
        if self.mode == 'device':
            print("[Receiver] Stopping thread and disconnecting from device...")
            try:
                self.explorer.stop_acquisition()
            except Exception:
                pass
            self.explorer.disconnect()
            
        elif self.mode == 'file':
            print("[Receiver] Stopping thread reading from file...")
            if self.explorer:
                self.explorer.stop_acquisition()



    def is_healthy(self, timeout=5.0):
        """Returns True if data has been received within the last `timeout` seconds."""
        if not self.connected or self.last_data_time is None:
            return False
        return (time.time() - self.last_data_time) < timeout

    def update_buffer(self, packet):
        '''
        Callback function to update the buffer with new EEG data packets.
        explorepy returns eeg shape (n_ch, n_samples) with a scalar timestamp.
        Each buffer entry is stored as (1 + n_ch,): [timestamp, ch0, ch1, ...].
        '''
        self.last_data_time = time.time()
        try:
            t, exg = packet.get_data()  # exg: (n_ch, n_samples), t: scalar
            exg = np.asarray(exg)       # (n_ch, n_samples)
            t   = float(np.asarray(t).ravel()[0])
            n_samples = exg.shape[1]

            with self.buffer_lock:
                for i in range(n_samples):
                    flat = np.empty(1 + exg.shape[0])
                    flat[0]  = t
                    flat[1:] = exg[:, i]
                    self.buffer.append(flat)
        except Exception as e:
            print(f"[Receiver] update_buffer error: {e}")


    def get_buffer_data(self):
        '''
        Get all data from the buffer as a numpy array.
        returns:
            data: np.array, shape (n_samples, n_ch)
        '''
        # Acquire lock to safely read from the buffer
        with self.buffer_lock:
            data = np.array(self.buffer)  # Convert deque to numpy array
        return data 
    


    def simulate_device(self):        
        '''
        Simulate Explore device using data file.
        args:
            self.data_path: str, path to data file
        returns:
            explore: MockExplore object
        '''
        
        # Inner MockPacket class
        class MockPacket:
            '''Mirrors explorepy packets: scalar timestamp, exg shaped (n_ch, n_samples).'''
            def __init__(self, timestamp, exg_data):
                self.timestamp = timestamp
                self.exg_data = exg_data

            def get_data(self):
                return self.timestamp, self.exg_data

        # Inner MockExplore class
        class MockExplore:
            def __init__(self, receiver_instance):
                self.receiver = receiver_instance
                self.data_path = receiver_instance.data_path
                self.running = False
                self.data = None
                self.load_data()
                
            def load_data(self):
                if self.data_path.endswith('.csv'):
                    df = pd.read_csv(self.data_path)
                    # Use 'TimeStamp' column if available, else first column
                    if 'TimeStamp' in df.columns:
                        timestamps = df['TimeStamp'].values
                        # ExG channels usually start after timestamp? 
                        # Based on "TimeStamp,ch1,ch2,ch3,ch4"
                        # We use columns 1: to be channels
                        exg = df.iloc[:, 1:].values
                    else:
                        print(f"Warning: 'TimeStamp' column not found in {self.data_path}. Assuming first col is timestamp.")
                        timestamps = df.iloc[:, 0].values
                        exg = df.iloc[:, 1:].values
                    
                    # Combine for easier iteration
                    self.times = timestamps
                    self.exg = exg
                    
                elif self.data_path.endswith('.npy'):
                    data = np.load(self.data_path)
                    # Assuming shape (N, n_ch+1) or similar protocol
                    self.times = data[:, 0]
                    self.exg = data[:, 1:]
                else:
                    raise ValueError("Unsupported file format. Use .csv or .npy")
                
                print(f"[MockExplore] Loaded {len(self.times)} samples.")


            def acquire(self):
                self.running = True
                idx = 0
                n_samples = len(self.times)
                # Calculate delays
                # Assuming first sample is t=0 for simulation, or respect absolute timestamps?
                # Best to respect relative time differences.
                
                print("[MockExplore] Starting acquisition simulation...")
                
                start_real_time = time.time()
                if n_samples > 0:
                    start_data_time = self.times[0] 
                
                while self.running and idx < n_samples:
                    current_data_time = self.times[idx]
                    
                    # Target elapsed time since start
                    target_elapsed = current_data_time - start_data_time
                    
                    # Current real elapsed time
                    real_elapsed = time.time() - start_real_time
                    
                    # Wait if we are ahead of schedule
                    wait_time = target_elapsed - real_elapsed
                    
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    # Create packet: one sample, shaped (n_ch, 1) like explorepy
                    packet = MockPacket(self.times[idx], self.exg[idx].reshape(-1, 1))
                    self.receiver.update_buffer(packet)
                    
                    idx += 1
                
                print("[MockExplore] End of file reached or stopped.")
                self.running = False

            def stop_acquisition(self):
                self.running = False

            def disconnect(self):
                pass
        
        return MockExplore(self)


if __name__ == "__main__":
    # def create_dummy_data(filename):
    #     n_samples = 70818 
    #     sampling_rate = 250 # Hz
    #     timestamps = np.arange(n_samples) / sampling_rate
        
    #     n_channels = 4
    #     data = np.zeros((n_samples, n_channels))
    #     for i in range(n_samples):
    #         for ch in range(n_channels):
    #             data[i, ch] = i + (ch * 1000) 
        
    #     df = pd.DataFrame(data, columns=[f'ch{i+1}' for i in range(n_channels)])
    #     df.insert(0, 'TimeStamp', timestamps)
    #     df.to_csv(filename, index=False)
    #     print(f"Created dummy CSV file: {filename} with {n_samples} samples.")

    # csv_filename = "test_data_temp.csv"
    # create_dummy_data(csv_filename)
    # csv_filename = "../data/MItest_24-01-27_ExG.csv"
    

    # Use a small buffer size to demonstrate circular buffer behavior
    # buffer_size = 1250
    # params = {
    #     'input_mode': 'file',
    #     'data_path': csv_filename,
    #     'buffer_size': buffer_size,
    #     'silent': False
    # }

    
    buffer_size = 1250
    params = {
        'input_mode': 'device',
        'buffer_size': buffer_size,
        'silent': False
    }

    try:
        print(f"Initializing EEG_Receiver with buffer_size={buffer_size}...")
        receiver = EEG_Receiver(params)
        
        print("Starting EEG Receiver thread...")
        receiver.start()
        
        start_monitor_time = time.time()
        duration = 999.0 
        
        print("\n=== Monitoring Buffer State (Circular Queue) ===")
        
        last_count = 0
        while time.time() - start_monitor_time < duration:
            # time.sleep(0.5)
            
            if not receiver.is_alive():
                print("Receiver thread finished.")
                break
                
            buffer_data = receiver.get_buffer_data()
            current_count = len(buffer_data)
            
            # print(f"\nTime: {time.time() - start_monitor_time:.1f}s | Buffer Size: {current_count}/{buffer_size}")

            # if current_count > 0:
                # print(f"  Oldest timestamp: {buffer_data[0][0]:.2f}")
                # if current_count > 1:
                #     print(f"  Newest timestamp: {buffer_data[-1][0]:.2f}")
                # print(f"  Newest Data: {buffer_data[-1][1:]}")
                
            last_count = current_count
        
        print("\nStop monitoring.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        if 'receiver' in locals() and receiver.is_alive():
            print("Stopping receiver...")
            receiver.stop()
            receiver.join()
            
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
            print(f"Removed temp file: {csv_filename}")
        

