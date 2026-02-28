import argparse
import time

from utils.load_config import load_config
from modules.eeg_receiver import EEG_Receiver
from modules.mi_predictor import MI_Predictor
from modules.evidence_accumulator import Evidence_Accumulator
from modules.wheelchair_controller import Wheelchair_Controller

def start_MI_Tracking(config_path):

    # --- Load Configuration ---
    config = load_config(config_path)

    # --- Initialize Modules ---
    receiver = EEG_Receiver(config['receiver_params']) 
    predictor = MI_Predictor(config['predictor_params'])
    accumulator = Evidence_Accumulator(config['accumulator_params'])
    controller = Wheelchair_Controller(config['control_params'])
    
    # sonic module

    # --- Start EEG Data Receiver ---
    receiver.run()

    # --- Main Control Loop  ---
    # Target control frequency: 20Hz (50ms interval)
    main_loop_interval = config.get('loop_interval', 0.05)
    print("[Main] Control loop started.")
    try:
        while True:
            loop_start = time.time()
            # Retrieves all buffered data from the receiver, shape: (n_samples, 1(timestamp) + n_ch), type: np.array
            data = receiver.get_buffer_data()

            if len(data) > 0:
                # preprocess and predict MI command
                raw_prediction = predictor.process_and_predict(data)

                if raw_prediction:
                    # Evidence Accumulation (Smoothing)
                    stable_cmd = accumulator.update(raw_prediction)

                    # Control Wheelchair based on stable command
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

            # Rate Limiting
            # Ensures the loop runs at ~20Hz to prevent CPU saturation
            elapsed = time.time() - loop_start
            sleep_time = main_loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
            print("\nStopping system...")
            receiver.stop()
            receiver.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the facemesh gaze tracker.")
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    # start MI tracking and control the wheelchair via EEG signals
    start_MI_Tracking(args.config_path)