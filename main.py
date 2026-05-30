import argparse
import time

from utils.load_config import load_config
from modules.eeg_receiver import EEG_Receiver
from modules.mi_predictor import MI_Predictor
from modules.evidence_accumulator import Evidence_Accumulator
from modules.wheelchair_controller import Wheelchair_Controller
from modules.gaze_receiver import Gaze_Receiver


def start_MI_Tracking(config_path):

    # --- Load Configuration ---
    config = load_config(config_path)

    # --- Initialize Modules ---
    receiver    = EEG_Receiver(config['receiver_params'])
    predictor   = MI_Predictor(config['predictor_params'])
    accumulator = Evidence_Accumulator(config['accumulator_params'])
    controller  = Wheelchair_Controller(config['control_params'])
    gaze        = Gaze_Receiver(config['gaze_params'])

    # --- Start Data Sources ---
    receiver.run()
    gaze.start()

    # --- Main Control Loop ---
    # Target control frequency: 20 Hz (50 ms interval)
    main_loop_interval = config.get('loop_interval', 0.05)
    print("[Main] Control loop started.")

    try:
        while True:
            loop_start = time.time()

            # ── MI: determine active / inactive gate state ──────────────────
            data = receiver.get_buffer_data()
            raw_mi = 'inactive'
            if len(data) > 0:
                raw_mi = predictor.process_and_predict(data)

            mi_state = accumulator.update(raw_mi)

            # ── Fused control: MI gate + Gaze direction ──────────────────────
            if mi_state != 'active':
                # MI says rest → stop regardless of where the user is looking
                print("[Main] MI inactive — stopped")
                controller.stop()
            else:
                # MI says grip → let gaze decide direction
                direction = gaze.get_direction()

                if direction == 'forward':
                    print("[Main] Forward")
                    controller.move_forward()
                elif direction == 'backward':
                    print("[Main] Backward")
                    controller.move_backward()
                elif direction == 'left':
                    print("[Main] Left")
                    controller.move_left()
                elif direction == 'right':
                    print("[Main] Right")
                    controller.move_right()
                else:
                    # direction == 'stop' or None (gaze still warming up)
                    print("[Main] Gaze centered / no face — stopped")
                    controller.stop()

            # ── Rate limiting: keep loop at ~20 Hz ──────────────────────────
            elapsed    = time.time() - loop_start
            sleep_time = main_loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Main] Stopping system...")
        gaze.stop()
        receiver.stop()
        receiver.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CereCe MI + Gaze wheelchair controller.")
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    start_MI_Tracking(args.config_path)
