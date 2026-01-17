from threading import Thread
from collections import deque
import numpy as np
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
        self.mode = params.get('input_mode', 'file')    # input from 'file' or 'device'
        self.running = False       
        self.daemon = True        
        self.buffer = deque(maxlen=params.get('buffer_size', 300)) # buffer shape: (buffer_size(timestamp), n_ch)
        self.buffer_lock = Thread.Lock()

        if self.mode == 'file':
            print(f"[Receiver] Input Mode: File")
            self.data_path = params.get('data_path', None)    
            self.explorer = self.simulate_device()
        
        elif self.mode == 'device':
            print(f"[Receiver] Input Mode: Device")
            self.device_name = params.get('device_name', 'Explore EEG Device')
            explorer = Explore()
            explorer.connect(device_name=self.device_name)

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
            print("[Receiver] Thread started. reading from file...")
            # Simulate reading data from file
            # Here you would implement the logic to read from your data file
            # and call self.update_buffer with the data packets.
            pass


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
        class MockExplore:
            def __init__(self, data_path, callback):
                self.data = np.load(data_path)  # Load data from file
                self.callback = callback
                self.index = 0
                self.running = False

            def acquire(self):
                self.running = True
                while self.running and self.index < self.data.shape[0]:
                    # Simulate data packet
                    t_vector = np.array([self.data[self.index, 0]])
                    exg_data = np.array([self.data[self.index, 1:]])
                    packet = MockPacket(t_vector, exg_data)
                    self.callback(packet)
                    self.index += 1

            def stop_acquisition(self):
                self.running = False

            def disconnect(self):
                pass
        

