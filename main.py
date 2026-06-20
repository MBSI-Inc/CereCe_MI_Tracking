import argparse
import time

from utils.load_config import load_config
from modules.eeg_receiver import EEG_Receiver
from modules.mi_predictor import MI_Predictor
from modules.evidence_accumulator import Evidence_Accumulator
from modules.wheelchair_controller import Wheelchair_Controller
from modules.gaze_control import GazeController, GazeControlError


def start_MI_Tracking(config_path):

    # --- Load Configuration ---
    config = load_config(config_path)

    # --- Initialize Modules ---
    receiver = EEG_Receiver(config['receiver_params'])
    predictor = MI_Predictor(config['predictor_params'])
    accumulator = Evidence_Accumulator(config['accumulator_params'])
    controller = Wheelchair_Controller(config['control_params'])

    # --- Start EEG Data Receiver ---
    receiver.run()

    # --- Main Control Loop  ---
    main_loop_interval = config.get('loop_interval', 0.05)
    print('[Main] Control loop started.')
    try:
        while True:
            loop_start = time.time()
            data = receiver.get_buffer_data()

            if len(data) > 0:
                raw_prediction = predictor.process_and_predict(data)
                if raw_prediction:
                    stable_cmd = accumulator.update(raw_prediction)
                    if stable_cmd == 'left':
                        print('Moving Left')
                        controller.move_left()
                    elif stable_cmd == 'right':
                        print('Moving Right')
                        controller.move_right()
                    else:
                        print('Stopping')
                        controller.stop()

            elapsed = time.time() - loop_start
            sleep_time = main_loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print('\nStopping system...')
        receiver.stop()
        receiver.join()


def start_Gaze_Tracking(config_path):
    config = load_config(config_path)
    gaze_controller = None

    try:
        gaze_controller = GazeController(config)
        gaze_controller.start(gaze_config_path=config.get('gaze_config_path', 'config.json'))
        print('[Gaze] Tracking started. SPACE = enable/disable movement. Q = quit.')

        main_loop_interval = config.get('loop_interval', 0.05)
        while True:
            loop_start = time.time()
            command = gaze_controller.update()
            if command is not None and command != 'paused':
                print(f'[Gaze] Command: {command}')

            key = gaze_controller.last_key
            if key == ord(' '):
                gaze_controller.gaze_enabled = not gaze_controller.gaze_enabled
                state = 'ENABLED' if gaze_controller.gaze_enabled else 'PAUSED'
                print(f'[Gaze] Movement {state}')
            elif key == ord('q'):
                print('[Gaze] Quit.')
                break

            elapsed = time.time() - loop_start
            sleep_time = main_loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print('\nStopping gaze tracking...')
    except GazeControlError as exc:
        print(f'[Gaze] Error: {exc}')
    finally:
        if gaze_controller is not None:
            gaze_controller.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the MI or gaze tracker.')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--mode', type=str, choices=['mi', 'gaze'], default='mi', help='Tracking mode to run')
    args = parser.parse_args()

    if args.mode == 'gaze':
        start_Gaze_Tracking(args.config_path)
    else:
        start_MI_Tracking(args.config_path)