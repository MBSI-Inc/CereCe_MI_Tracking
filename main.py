import argparse
import time
import cv2

from utils.load_config import load_config
from modules.eeg_receiver import EEG_Receiver
from modules.mi_predictor import MI_Predictor
from modules.evidence_accumulator import Evidence_Accumulator
from modules.wheelchair_controller import Wheelchair_Controller
from modules.gaze_receiver import Gaze_Receiver


def start_MI_Tracking(config_path):

    # --- Load Configuration ---
    config = load_config(config_path)

    # --- Module Enable/Disable Flags ---
    modules_config = config.get('modules', {})
    eeg_enabled = modules_config.get('eeg_receiver', True)
    gaze_enabled = modules_config.get('gaze_receiver', True)
    control_enabled = modules_config.get('wheelchair_controller', True)

    # --- Initialize Modules ---
    receiver    = EEG_Receiver(config['receiver_params']) if eeg_enabled else None
    predictor   = MI_Predictor(config['predictor_params']) if eeg_enabled else None
    accumulator = Evidence_Accumulator(config['accumulator_params']) if eeg_enabled else None
    controller  = Wheelchair_Controller(config['control_params']) if control_enabled else None
    gaze        = Gaze_Receiver(config['gaze_params']) if gaze_enabled else None
    show_camera_ui = config.get('show_camera_ui', True)

    # --- Start Data Sources ---
    if receiver is not None:
        receiver.start()
    if gaze is not None:
        gaze.start()

    # --- Main Control Loop ---
    # Target control frequency: 20 Hz (50 ms interval)
    main_loop_interval = config.get('loop_interval', 0.05)
    print("[Main] Control loop started.")

    try:
        while True:
            loop_start = time.time()

            # ── MI: determine active / inactive gate state ──────────────────
            if eeg_enabled:
                data = receiver.get_buffer_data()
                raw_mi = 'inactive'
                if len(data) > 0:
                    raw_mi = predictor.process_and_predict(data)

                mi_state = accumulator.update(raw_mi)
            else:
                # EEG disabled: always treat as 'active' to allow gaze-only control
                mi_state = 'active'
                print("[Main] EEG disabled - gaze-only mode")

            # ── Fused control: MI gate + Gaze direction ──────────────────────
            direction = None
            if not control_enabled:
                # Controller disabled: just print direction without moving
                if gaze_enabled:
                    direction = gaze.get_direction()
                    print(f"[Main] [GAZE-ONLY] Direction: {direction}")
            elif mi_state != 'active':
                # MI says rest → stop regardless of where the user is looking
                print("[Main] MI inactive — stopped")
                controller.stop()
            else:
                # MI says grip → let gaze decide direction
                direction = gaze.get_direction() if gaze_enabled else 'stop'

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

            if gaze_enabled and show_camera_ui:
                frame = gaze.get_frame()
                if frame is not None:
                    direction_text = direction if direction is not None else 'warming up'
                    cv2.putText(frame, f'Direction: {direction_text}', (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Camera UI', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

            # ── Rate limiting: keep loop at ~20 Hz ──────────────────────────
            elapsed    = time.time() - loop_start
            sleep_time = main_loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Main] Stopping system...")
        if gaze is not None:
            gaze.stop()
        if receiver is not None:
            receiver.stop()
            receiver.join()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CereCe MI + Gaze wheelchair controller.")
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    start_MI_Tracking(args.config_path)
