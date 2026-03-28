from threading import Thread, Lock
from collections import deque
import numpy as np
import pandas as pd
import time
import os
from explorepy import Explore
from explorepy.stream_processor import TOPICS


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


        if self.mode == 'file':
            print(f"[Receiver] Input Mode: File")
            self.data_path = params.get('data_path', None)    
            self.explorer = self.simulate_device()
        
        elif self.mode == 'device':
            print(f"[Receiver] Input Mode: Device")
            self.device_name = params.get('device_name', 'Explore_842F')
            self.explorer = Explore()
            self.explorer.connect(device_name=self.device_name)

            ## TODO: add timestamp verification and impedance verification here
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
            try:
                self.explorer.acquire()
            except Exception as e:
                print(f"Error: {e}")
                self.running = False

        elif self.mode == 'file':
            print(f"[Receiver] Thread started. reading from file: {self.data_path}...")
            if self.explorer:
                 self.explorer.acquire()


    def stop(self):
        self.running = False
        if self.mode == 'device':
            print("[Receiver] Stopping thread and disconnecting from device...")
            self.explorer.stop_acquisition()
            self.explorer.disconnect()
            
        elif self.mode == 'file':
            print("[Receiver] Stopping thread reading from file...")
            # Implement any necessary cleanup for file reading here
            pass



    def update_buffer(self, packet):
        '''
        Callback function to update the buffer with new EEG data packets.
        args:
            packet: explorepy data packet
        '''
        # get data from the packet
        t_vector, exg_data = packet.get_data() # t_vector: (N,), exg_data: (N, n_ch)
        print("t_vector" + t_vector)
        # Acquire lock to safely update the buffer
        with self.buffer_lock:
            # put each sample into the buffer queue
            for i in range(exg_data.shape[0]):
                time_stamp = t_vector[i]
                sample = exg_data[i, :]
                flat_sample = np.concatenate(([time_stamp], sample)) # shape: (1 + n_ch,)
                self.buffer.append(flat_sample)  


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
            def __init__(self, t_vector, exg_data):
                self.t_vector = t_vector
                self.exg_data = exg_data
            
            def get_data(self):
                return self.t_vector, self.exg_data

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
                    
                    # Create packet
                    t_vec = np.array([self.times[idx]])
                    # Ensure shape is (1, n_ch)
                    d_vec = self.exg[idx].reshape(1, -1)
                    
                    packet = MockPacket(t_vec, d_vec)
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
            
            print(f"\nTime: {time.time() - start_monitor_time:.1f}s | Buffer Size: {current_count}/{buffer_size}")
            
            if current_count > 0:
                print(f"  Oldest timestamp: {buffer_data[0][0]:.2f}")
                if current_count > 1:
                    print(f"  Newest timestamp: {buffer_data[-1][0]:.2f}")
                # print last sample content to verify data
                print(f"  Newest Data: {buffer_data[-1][1:]}")
                
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
        

