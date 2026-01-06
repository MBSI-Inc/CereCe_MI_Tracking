import argparse
import time
from utils.eeg_receiver import EEG_Receiver
from utils.mi_predictor import MI_Predictor
from utils.evidence_accumulator import Evidence_Accumulator
from utils.wheelchair_controller import Wheelchair_Controller

def start_MI_Tracking(config):
    '''
    Orchestrates the main real-time BCI control pipeline.

    This function implements an asynchronous "Producer-Consumer" architecture:
    1. Hardware Layer (Background Thread): The EEG_Receiver captures raw data 
       at high frequency (250Hz) to prevent packet loss.
    2. Logic Layer (Main Loop): This loop processes data, predicts commands, 
       and updates the wheelchair controller at a stable control rate (~20Hz).

    Key Features:
        - Decoupling: Separates hardware data ingestion from signal processing.
        - Smoothing: Uses Evidence Accumulation to filter out signal jitter.
        - Rate Limiting: Maintains a consistent loop interval to manage CPU load.

    Args:
        config (dict): Configuration parameters for all subsystems.
    '''
    # --- Constants ---
    # Target control frequency: 20Hz (50ms interval)
    LOOP_INTERVAL = 0.05 

    # --- Initialization ---
    # Receiver handles high-frequency (250Hz) hardware IO in a background thread
    receiver = EEG_Receiver(config) 
    predictor = MI_Predictor(config)
    accumulator = Evidence_Accumulator(config)
    controller = Wheelchair_Controller(config)

    print("Starting EEG Stream...")
    receiver.start_stream()

    print("System Ready. Entering Control Loop...")

    # --- Main Control Loop (~20Hz) ---
    while True:
        loop_start = time.time()

        # 1. Fetch Data (Consumer)
        # Retrieves all buffered data accumulated since the last loop iteration
        data = receiver.get_latest_data()

        if len(data) > 0:
            # 2. Predict (Inference)
            # Returns None if buffer isn't full yet, or a probability distribution
            raw_prediction = predictor.process_and_predict(data)

            if raw_prediction:
                # 3. Stabilize (Evidence Accumulation)
                # Integrates probability over time to produce a stable command
                stable_cmd = accumulator.update(raw_prediction)

                # 4. Actuate (Control)
                if stable_cmd == 'left':
                    print("Moving Left")
                    controller.move_left()
                elif stable_cmd == 'right':
                    print("Moving Right")
                    controller.move_right()
                else:
                    # Default safe state: Stop if command is 'stop' or uncertain
                    print("Stopping")
                    controller.stop()

        # 5. Rate Limiting
        # Ensures the loop runs at ~20Hz to prevent CPU saturation
        elapsed = time.time() - loop_start
        sleep_time = LOOP_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the facemesh gaze tracker.")
    parser.add_argument('--config', type=str, default='config.json', help='Config file')
    args = parser.parse_args()

    # start MI tracking and control the wheelchair via EEG signals
    start_MI_Tracking(args.config)