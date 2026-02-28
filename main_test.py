import argparse
import time
import os
import sys

# Ensure modules can be imported
sys.path.append(os.getcwd())

from utils.load_config import load_config
from modules.eeg_receiver import EEG_Receiver
from modules.mi_predictor import MI_Predictor
from modules.evidence_accumulator import Evidence_Accumulator
from modules.wheelchair_controller import Wheelchair_Controller

def start_MI_Tracking_Test(config_path):

    # --- Load Configuration ---
    config = load_config(config_path)

    # --- Override for Test ---
    print("--- Test Mode ---")
    config['receiver_params']['input_mode'] = 'file'
    # Use the provided CSV file
    test_file = 'data/MItest_24-01-27_ExG.csv'
    if os.path.exists(test_file):
        config['receiver_params']['data_path'] = test_file
    else:
        print(f"Warning: Test file {test_file} not found. Using config default.")
    
    print(f"Receiver Input: {config['receiver_params']['input_mode']}")
    print(f"Data Path: {config['receiver_params']['data_path']}")

    # --- Initialize Modules ---
    receiver = EEG_Receiver(config['receiver_params']) 
    predictor = MI_Predictor(config['predictor_params'])
    accumulator = Evidence_Accumulator(config['accumulator_params'])
    controller = Wheelchair_Controller(config['control_params'])

    # --- Start EEG Data Receiver ---
    receiver.start() # Start the thread

    # --- Main Control Loop  ---
    # Target control frequency: 20Hz (50ms interval)
    main_loop_interval = config.get('loop_interval', 0.05)
    print("[Main Test] Control loop started.")
    
    try:
        while True:
            loop_start = time.time()
            
            # --- Module 1: Receiver Status ---
            # Retrieves all buffered data from the receiver
            data = receiver.get_buffer_data()
            buffer_status = f"Buffer Size: {len(data)}"
            
            prediction_status = "No Prediction"
            accumulator_status = "No Update"
            controller_status = "No Action"

            if len(data) > 0:
                # --- Module 2: Predictor Status ---
                # preprocess and predict MI command
                raw_prediction = predictor.process_and_predict(data)
                
                if raw_prediction:
                    prediction_status = f"Raw: {raw_prediction}"
                    
                    # --- Module 3: Accumulator Status ---
                    # Evidence Accumulation (Smoothing)
                    stable_cmd = accumulator.update(raw_prediction)
                    accumulator_status = f"Evidence: {accumulator.evidence}, Stable: {stable_cmd}"

                    # --- Module 4: Controller Status ---
                    # Control Wheelchair based on stable command
                    if stable_cmd == 'left':
                        controller.move_left()
                        controller_status = "Moving Left"
                    elif stable_cmd == 'right':
                        controller.move_right()
                        controller_status = "Moving Right"
                    else:
                        controller.stop()
                        controller_status = "Stopping"
            
            # Print Integration Status
            print(f"[Status] {buffer_status} | {prediction_status} | {accumulator_status} | {controller_status}")

            # Rate Limiting
            elapsed = time.time() - loop_start
            sleep_time = main_loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Check if receiver is still running (end of file)
            if not receiver.running and len(data) == 0:
                print("[Main Test] Input stream ended.")
                break

    except KeyboardInterrupt:
            print("\nStopping system...")
            
    finally:
        receiver.stop()
        receiver.join()
        print("[Main Test] Test finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MI Tracking Integration Test.")
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    start_MI_Tracking_Test(args.config_path)
