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
from modules.ultrasonic_receiver import Ultrasonic_Receiver

EEG_WARMUP_SECONDS = 5.0
EEG_HEALTH_TIMEOUT = 5.0

_KEY_MAP = {
    ord('w'): 'forward',
    ord('s'): 'backward',
    ord('a'): 'left',
    ord('d'): 'right',
}

_MODE_COLORS = {
    'EEG+GAZE':    (255, 255, 0),
    'EEG+KB':      (0, 200, 255),
    'GAZE+KB':     (255, 0, 255),
    'KEYBOARD':    (0, 255, 255),
    'KB OVERRIDE': (0, 128, 255),
    'HEADLESS':    (180, 180, 180),
}


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _dispatch(controller, command):
    """Send one already-deduplicated command to the wheelchair controller."""
    if command == 'forward':
        controller.move_forward()
    elif command == 'backward':
        controller.move_backward()
    elif command == 'left':
        controller.move_left()
    elif command == 'right':
        controller.move_right()
    else:
        controller.stop()


def start_MI_Tracking(config_path):

    config = load_config(config_path)

    modules_config  = config.get('modules', {})
    eeg_enabled     = modules_config.get('eeg_receiver', True)
    eeg_requested   = eeg_enabled   # captured before any runtime fallback
    gaze_enabled    = modules_config.get('gaze_receiver', True)
    control_enabled = modules_config.get('wheelchair_controller', True)
    sonar_enabled   = modules_config.get('ultrasonic_receiver', False)

    headless               = _as_bool(config.get('headless', False))
    show_camera_ui         = _as_bool(config.get('show_camera_ui', True))
    run_duration           = float(config.get('run_duration', 0) or 0)
    exit_on_source_end     = _as_bool(config.get('exit_on_source_end', False))
    command_keepalive      = float(config.get('command_keepalive', 0.5))
    keyboard_override_hold = float(config.get('keyboard_override_hold', 0.25))
    main_loop_interval     = config.get('loop_interval', 0.05)

    # ── Module construction ───────────────────────────────────────────────
    receiver = predictor = accumulator = None
    if eeg_enabled:
        try:
            receiver    = EEG_Receiver(config['receiver_params'])
            predictor   = MI_Predictor(config['predictor_params'])
            accumulator = Evidence_Accumulator(config['accumulator_params'])
        except Exception as e:
            # Drop every partially-built EEG object so nothing half-started
            # is left behind (a live receiver thread with no consumer).
            print(f"[Main] EEG init failed: {e} — EEG module disabled for this run.")
            receiver = predictor = accumulator = None
            eeg_enabled = False

    controller = Wheelchair_Controller(config['control_params']) if control_enabled else None
    gaze       = Gaze_Receiver(config['gaze_params']) if gaze_enabled else None

    sonar = None
    if sonar_enabled:
        sonar = Ultrasonic_Receiver(config.get('ultrasonic_params', {}))
        if not sonar.available:
            sonar = None
            print("[Main] Ultrasonic module disabled (bleak unavailable) — obstacle guard off.")

    if receiver is not None:
        receiver.start()
    if gaze is not None:
        gaze.start()
    if sonar is not None:
        sonar.start()

    # ── EEG gate warmup ───────────────────────────────────────────────────
    # 'eeg_gate' means "the MI signal is usable as the drive gate". It is
    # deliberately independent of where steering comes from: gaze, keyboard,
    # or both, keep working whether or not the gate is live.
    eeg_gate = False
    if eeg_enabled and receiver is not None:
        print(f"[Main] Waiting {EEG_WARMUP_SECONDS}s for EEG warmup...")
        time.sleep(EEG_WARMUP_SECONDS)
        eeg_gate = receiver.is_healthy(EEG_HEALTH_TIMEOUT)
        if eeg_gate:
            print("[Main] EEG healthy — MI gate is live.")
        else:
            print("[Main] EEG not collecting data — MI gate unavailable; "
                  "steering continues without it.")
    elif eeg_requested:
        print("[Main] EEG unavailable (init failed) — MI gate unavailable.")

    # ── Initial drive state ───────────────────────────────────────────────
    start_drive = config.get('start_drive_enabled', 'auto')
    if isinstance(start_drive, str) and start_drive.strip().lower() == 'auto':
        # Only auto-arm when EEG was never requested (explicit gaze/keyboard
        # bench run). A requested-but-broken EEG must NOT start the motors.
        drive_enabled = not eeg_requested
    else:
        drive_enabled = _as_bool(start_drive)
    print(f"[Main] Drive starts {'ENABLED' if drive_enabled else 'DISABLED'} "
          f"(start_drive_enabled={start_drive}).")

    # ── Loop state ────────────────────────────────────────────────────────
    keyboard_direction      = None   # None = no WASD key held
    keyboard_override_until = 0.0
    prev_mi_gate       = 'inactive'
    reverse_gear       = False       # False = forward, True = reverse (blink toggles)
    prev_blink         = False
    force_kb_steering  = False       # 'g' sticky keyboard override
    last_command       = None
    last_command_time  = 0.0
    sonar_warned       = False
    loop_start_time    = time.time()

    print("[Main] Control loop started.")
    if not control_enabled:
        print("[Main] wheelchair_controller disabled — DRY RUN (no motor commands).")
    if headless:
        print("[Main] Headless mode: no window and no keyboard input. "
              "Ctrl+C to stop.")
    else:
        print("[Main] Keys: WASD=steer  "
              + ("" if eeg_gate else "SPACE=drive  ")
              + ("G=keyboard override  " if gaze is not None else "")
              + "Q=quit"
              + ("  (drive is toggled by the MI gate)" if eeg_gate else ""))

    try:
        while True:
            loop_start = time.time()

            # ── MI gate ──────────────────────────────────────────────────
            if eeg_enabled and receiver is not None:
                data = receiver.get_buffer_data()
                raw_mi = predictor.process_and_predict(data) if len(data) > 0 else 'none'
                mi_gate = accumulator.update(raw_mi)

                # Toggle drive on rising edge (inactive → active)
                if mi_gate == 'active' and prev_mi_gate == 'inactive':
                    drive_enabled = not drive_enabled
                    print(f"[Main] Drive {'ENABLED' if drive_enabled else 'DISABLED'} (MI gate)")
                prev_mi_gate = mi_gate

                if receiver.is_healthy(EEG_HEALTH_TIMEOUT):
                    if not eeg_gate:
                        print("[Main] EEG signal detected — MI gate live "
                              "(drive disarmed; MI toggles it).")
                        drive_enabled = False
                        prev_mi_gate = 'inactive'
                    eeg_gate = True
                else:
                    if eeg_gate:
                        print("[Main] EEG data loss — MI gate unavailable; "
                              "steering continues without it.")
                    eeg_gate = False
                    prev_mi_gate = 'inactive'
            else:
                eeg_gate = False

            # End-of-replay / end-of-stream termination
            if exit_on_source_end and receiver is not None and receiver.finished:
                print("[Main] EEG source finished — stopping.")
                break

            # ── Direction source: held key wins, then gaze, else stop ────
            if force_kb_steering:
                direction = keyboard_direction or 'stop'
                steering_source = 'keyboard (override)'
            elif keyboard_direction is not None:
                direction = keyboard_direction
                steering_source = 'keyboard'
            elif gaze is not None:
                gaze_direction = gaze.get_direction()
                if gaze_direction is None:
                    direction = 'stop'
                    steering_source = 'gaze (warming up)'
                else:
                    direction = gaze_direction
                    steering_source = 'gaze'
            else:
                direction = 'stop'
                steering_source = 'none'

            # ── Blink gear toggle (needs gaze, independent of steering) ──
            if gaze is not None:
                blink = gaze.get_blink()
                if blink and not prev_blink:
                    reverse_gear = not reverse_gear
                    print(f"[Main] Gear: {'REVERSE' if reverse_gear else 'FORWARD'} (blink)")
                prev_blink = blink

            # Flip forward↔backward when in reverse gear; left/right unchanged
            if reverse_gear:
                if direction == 'forward':
                    direction = 'backward'
                elif direction == 'backward':
                    direction = 'forward'

            # ── Obstacle guard ────────────────────────────────────────────
            # Ultrasonic sensor overrides blocked directions with 'stop'
            obstacle = sonar.is_obstacle() if sonar is not None else False
            if sonar is not None:
                direction = sonar.filter_direction(direction, obstacle)
                if (not sonar_warned and not sonar.is_healthy()
                        and (time.time() - loop_start_time) > 10.0):
                    print("[Main] WARNING: ultrasonic sensor is not reporting — "
                          "obstacle guard inactive.")
                    sonar_warned = True

            # ── Control (deduplicated + keepalive) ────────────────────────
            command = direction if drive_enabled else 'stop'
            now = time.time()
            due = (command != last_command) or (now - last_command_time) >= command_keepalive
            if due:
                if not control_enabled:
                    print(f"[Main] [DRY RUN] Drive: {'ON' if drive_enabled else 'OFF'} "
                          f"| Dir: {command} | Steering: {steering_source}")
                else:
                    _dispatch(controller, command)
                    if command != last_command:
                        print(f"[Main] Command: {command} (steering: {steering_source})")
                last_command = command
                last_command_time = now

            # ── UI & keyboard ─────────────────────────────────────────────
            if not headless:
                frame = gaze.get_frame() if (show_camera_ui and gaze is not None) else None
                if frame is None:
                    frame = np.zeros((200, 640, 3), dtype=np.uint8)
                    if show_camera_ui:
                        cv2.putText(frame, 'Camera warming up...', (20, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

                if force_kb_steering:
                    base_label = "KB OVERRIDE"
                elif eeg_gate and gaze is not None:
                    base_label = "EEG+GAZE"
                elif eeg_gate:
                    base_label = "EEG+KB"
                elif gaze is not None:
                    base_label = "GAZE+KB"
                else:
                    base_label = "KEYBOARD"
                mode_label = base_label + ('+SONAR' if sonar is not None else '')
                color_mode = _MODE_COLORS.get(base_label, (200, 200, 200))

                cv2.putText(frame, f'Mode: {mode_label}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_mode, 2, cv2.LINE_AA)
                cv2.putText(frame, f'Direction: {direction}', (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                drive_label = "DRIVE: ON" if drive_enabled else "DRIVE: OFF"
                drive_color = (0, 255, 0) if drive_enabled else (0, 0, 255)
                gear_label  = "GEAR: REV" if reverse_gear else "GEAR: FWD"
                gear_color  = (0, 165, 255) if reverse_gear else (255, 255, 255)
                cv2.putText(frame, drive_label, (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, drive_color, 2, cv2.LINE_AA)
                cv2.putText(frame, gear_label, (20, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, gear_color, 2, cv2.LINE_AA)

                if sonar is not None:
                    dist = sonar.get_distances()
                    if not sonar.is_healthy() or dist is None:
                        sonar_label, sonar_color = "SONAR: --", (128, 128, 128)
                    elif obstacle:
                        sonar_label = f"OBSTACLE  L={dist[0]:.0f} R={dist[1]:.0f}cm"
                        sonar_color = (0, 0, 255)
                    else:
                        sonar_label = f"SONAR: L={dist[0]:.0f} R={dist[1]:.0f}cm"
                        sonar_color = (0, 255, 0)
                    cv2.putText(frame, sonar_label, (frame.shape[1] - 330, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, sonar_color, 2, cv2.LINE_AA)

                hints = "WASD=steer  Q=quit"
                if not eeg_gate:
                    hints = "SPACE=drive  " + hints
                if gaze is not None:
                    hints += "  G=keyboard override"
                cv2.putText(frame, hints, (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

                cv2.imshow('Camera UI', frame)

                # Read key — hold a WASD key to steer; release returns to gaze.
                # The latch keeps working on platforms where cv2.waitKey does
                # not emit auto-repeat events while a key is held.
                key = cv2.waitKey(1) & 0xFF
                now = time.time()
                if key == ord('q'):
                    raise KeyboardInterrupt
                # Closing the window with X also stops everything
                try:
                    if cv2.getWindowProperty('Camera UI', cv2.WND_PROP_VISIBLE) < 1:
                        raise KeyboardInterrupt
                except cv2.error:
                    pass

                direction_key = _KEY_MAP.get(key)
                if direction_key is not None:
                    keyboard_direction = direction_key
                    keyboard_override_until = now + keyboard_override_hold
                elif key == 0xFF and now > keyboard_override_until:
                    keyboard_direction = None

                if key == ord(' '):
                    if eeg_gate:
                        print("[Main] Drive is toggled by the MI gate; spacebar ignored.")
                    else:
                        drive_enabled = not drive_enabled
                        print(f"[Main] Drive {'ENABLED' if drive_enabled else 'DISABLED'} (spacebar)")
                if key == ord('g'):
                    force_kb_steering = not force_kb_steering
                    print(f"[Main] Steering: "
                          f"{'KEYBOARD OVERRIDE' if force_kb_steering else 'AUTO (held key > gaze)'} (g key)")

            # ── Termination & rate limiting ───────────────────────────────
            if run_duration > 0 and (time.time() - loop_start_time) >= run_duration:
                print(f"[Main] Reached run_duration ({run_duration}s) — stopping.")
                break

            elapsed = time.time() - loop_start
            sleep_time = main_loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Main] Stopping system...")
    finally:
        # Motors are stopped on EVERY exit path (q, window close, Ctrl+C,
        # exception, duration/source end) so a live chair never keeps its
        # last commanded target after the loop dies.
        if controller is not None:
            try:
                controller.stop()
                print("[Main] Motors stopped.")
            except Exception as e:
                print(f"[Main] Failed to stop motors: {e}")
        if gaze is not None:
            gaze.stop()
        if sonar is not None:
            sonar.stop()
        if receiver is not None:
            receiver.stop()
            receiver.join(timeout=2.0)
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CereCe MI + Gaze wheelchair controller.")
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Config file path')
    args = parser.parse_args()

    start_MI_Tracking(args.config_path)
