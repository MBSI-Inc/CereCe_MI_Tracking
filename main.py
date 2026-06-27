import argparse
import time
import cv2
import numpy as np

from utils.load_config import load_config
from modules.eeg_receiver import EEG_Receiver
from modules.mi_predictor import MI_Predictor
from modules.evidence_accumulator import Evidence_Accumulator
from modules.wheelchair_controller import Wheelchair_Controller
from modules.gaze_receiver import Gaze_Receiver

EEG_WARMUP_SECONDS = 5.0
EEG_HEALTH_TIMEOUT = 5.0

_KEY_MAP = {
    ord('w'): 'forward',
    ord('s'): 'backward',
    ord('a'): 'left',
    ord('d'): 'right',
}


def start_MI_Tracking(config_path):

    config = load_config(config_path)

    modules_config = config.get('modules', {})
    eeg_enabled    = modules_config.get('eeg_receiver', True)
    gaze_enabled   = modules_config.get('gaze_receiver', True)
    control_enabled = modules_config.get('wheelchair_controller', True)

    receiver = predictor = accumulator = None
    if eeg_enabled:
        try:
            receiver    = EEG_Receiver(config['receiver_params'])
            predictor   = MI_Predictor(config['predictor_params'])
            accumulator = Evidence_Accumulator(config['accumulator_params'])
        except Exception as e:
            print(f"[Main] EEG init failed: {e} — falling back to keyboard control.")
            eeg_enabled = False

    controller = Wheelchair_Controller(config['control_params']) if control_enabled else None
    gaze       = Gaze_Receiver(config['gaze_params']) if gaze_enabled else None
    show_camera_ui = config.get('show_camera_ui', True)

    if receiver is not None:
        receiver.start()
    if gaze is not None:
        gaze.start()

    # EEG health check after warmup
    gaze_only_fallback = not eeg_enabled
    if eeg_enabled:
        print(f"[Main] Waiting {EEG_WARMUP_SECONDS}s for EEG warmup...")
        time.sleep(EEG_WARMUP_SECONDS)
        if not receiver.is_healthy():
            print("[Main] EEG not collecting data — falling back to keyboard control.")
            gaze_only_fallback = True
        else:
            print("[Main] EEG healthy. Running in EEG+Gaze mode.")

    # Keyboard direction, updated each frame from cv2.waitKey
    keyboard_direction = 'stop'

    # MI toggle state — each confirmed MI detection flips drive on/off
    drive_enabled = False
    prev_mi_gate = 'inactive'

    main_loop_interval = config.get('loop_interval', 0.05)
    print("[Main] Control loop started.")
    if gaze_only_fallback:
        print("[Main] Mode: KEYBOARD FALLBACK. SPACE = toggle drive, WASD = steer, Q = quit.")

    try:
        while True:
            loop_start = time.time()

            # ── MI gate ──────────────────────────────────────────────────────
            if not gaze_only_fallback:
                data = receiver.get_buffer_data()
                raw_mi = predictor.process_and_predict(data) if len(data) > 0 else 'none'
                mi_gate = accumulator.update(raw_mi)

                # Toggle drive on rising edge (inactive → active)
                if mi_gate == 'active' and prev_mi_gate == 'inactive':
                    drive_enabled = not drive_enabled
                    print(f"[Main] Drive {'ENABLED' if drive_enabled else 'DISABLED'}")
                prev_mi_gate = mi_gate

                if not receiver.is_healthy(EEG_HEALTH_TIMEOUT):
                    print("[Main] EEG data loss — switching to keyboard control.")
                    gaze_only_fallback = True
                    drive_enabled = False
                    prev_mi_gate = 'inactive'
            else:
                # Recover if EEG becomes healthy after a warmup failure or transient loss
                if eeg_enabled and receiver is not None and receiver.is_healthy(EEG_HEALTH_TIMEOUT):
                    print("[Main] EEG signal detected — switching to EEG+Gaze mode.")
                    gaze_only_fallback = False
                    drive_enabled = False
                    prev_mi_gate = 'inactive'

            # ── Direction source ──────────────────────────────────────────────
            if gaze_only_fallback or not gaze_enabled:
                direction = keyboard_direction
            else:
                direction = gaze.get_direction()

            # ── Control ───────────────────────────────────────────────────────
            if not control_enabled:
                print(f"[Main] [DRY RUN] Drive: {'ON' if drive_enabled else 'OFF'} | Dir: {direction}")
            elif not drive_enabled:
                controller.stop()
            else:
                if direction == 'forward':
                    controller.move_forward()
                elif direction == 'backward':
                    controller.move_backward()
                elif direction == 'left':
                    controller.move_left()
                elif direction == 'right':
                    controller.move_right()
                else:
                    controller.stop()

            # ── Camera UI & keyboard reading ──────────────────────────────────
            if show_camera_ui:
                frame = gaze.get_frame() if gaze_enabled else None

                if frame is None:
                    frame = np.zeros((200, 500, 3), dtype=np.uint8)
                    cv2.putText(frame, 'Camera warming up...', (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

                if gaze_only_fallback:
                    mode_label, color_mode = "KEYBOARD", (0, 255, 255)
                elif gaze_enabled:
                    mode_label, color_mode = "EEG+GAZE", (255, 255, 0)
                else:
                    mode_label, color_mode = "EEG+KB", (0, 200, 255)
                drive_label = "DRIVE: ON" if drive_enabled else "DRIVE: OFF"
                drive_color = (0, 255, 0) if drive_enabled else (0, 0, 255)
                cv2.putText(frame, f'Mode: {mode_label}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_mode, 2, cv2.LINE_AA)
                cv2.putText(frame, f'Direction: {direction}', (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                if not gaze_only_fallback:
                    cv2.putText(frame, drive_label, (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, drive_color, 2, cv2.LINE_AA)
                if gaze_only_fallback or not gaze_enabled:
                    cv2.putText(frame, 'SPACE = toggle   WASD = steer   Q = quit', (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

                cv2.imshow('Camera UI', frame)

                # Read key — hold key to move, release to stop; space toggles drive
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt
                if key == ord(' '):
                    drive_enabled = not drive_enabled
                    print(f"[Main] Drive {'ENABLED' if drive_enabled else 'DISABLED'} (spacebar)")
                keyboard_direction = _KEY_MAP.get(key, 'stop')

            # ── Rate limiting ─────────────────────────────────────────────────
            elapsed = time.time() - loop_start
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
