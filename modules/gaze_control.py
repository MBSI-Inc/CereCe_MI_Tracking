import os
import time
import cv2

from modules.wheelchair_controller import Wheelchair_Controller

try:
    from Cerebruh_Gaze_Tracking.src.facemesh_gazetracker import GazeTracker
except Exception as exc:
    GazeTracker = None
    _GazeImportError = exc


class GazeControlError(Exception):
    pass


class GazeController:
    """Maps gaze tracker output into wheelchair movement commands."""

    def __init__(self, config):
        self.config = config
        self.controller = Wheelchair_Controller(config['control_params'])
        self.gaze_config = config.get('gaze_params', {}) or {}
        self.horizontal_threshold = float(self.gaze_config.get('horizontal_threshold', 50.0))
        self.horizontal_deadband = float(self.gaze_config.get('horizontal_deadband', 20.0))
        self.display_frame = bool(self.gaze_config.get('display_frame', True))
        self.face_required = bool(self.gaze_config.get('face_required', True))
        self.gaze_enabled = False
        self.last_key = -1
        self._reverse = False
        self._last_blinked_state = None
        self.gaze_tracker = None

        if GazeTracker is None:
            raise GazeControlError(
                f"GazeTracker import failed: {_GazeImportError}. "
                "Make sure the Cerebruh_Gaze_Tracking submodule is initialized and available."
            )

    def start(self, gaze_config_path=None):
        if gaze_config_path is None:
            gaze_config_path = self.gaze_config.get('config_path', 'config.json')

        gaze_config_path = os.path.expanduser(gaze_config_path)
        candidate_paths = [gaze_config_path]

        if not os.path.isabs(gaze_config_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            gaze_root = os.path.join(project_root, 'Cerebruh_Gaze_Tracking')
            candidate_paths.append(os.path.join(gaze_root, gaze_config_path))
            candidate_paths.append(os.path.join(gaze_root, 'src', gaze_config_path))

        existing_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if existing_path is None:
            raise GazeControlError(
                f"Gaze config file not found: {gaze_config_path}. "
                f"Tried: {', '.join(candidate_paths)}"
            )

        self.gaze_tracker = GazeTracker(existing_path)
        self.gaze_tracker.start()
        time.sleep(0.5)

    def stop(self):
        if self.gaze_tracker is not None:
            try:
                self.gaze_tracker.stop()
            except Exception:
                pass
        self.controller.stop()
        cv2.destroyAllWindows()

    def update(self):
        if self.gaze_tracker is None:
            raise GazeControlError("GazeTracker is not started.")

        gaze_output = self.gaze_tracker.get_gaze_async()
        if gaze_output is None:
            return None

        command = self._process_gaze_output(gaze_output)

        if self.display_frame:
            try:
                frame = self.gaze_tracker.current_control_frame.copy()
                self._draw_status_overlay(frame, command)
                cv2.imshow("Gazetracker frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty("Gazetracker frame", cv2.WND_PROP_VISIBLE) < 1:
                    self.last_key = ord('q')
                else:
                    self.last_key = key
            except Exception:
                self.last_key = -1

        return command

    def _draw_status_overlay(self, frame, command=None):
        h, w = frame.shape[:2]

        # Semi-transparent black banner across the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        if not self.gaze_enabled:
            label, color = "PAUSED", (0, 0, 255)
        elif command == 'left':
            label, color = "LEFT", (255, 200, 0)
        elif command == 'right':
            label, color = "RIGHT", (255, 200, 0)
        elif command == 'backward':
            label, color = "REVERSE", (0, 140, 255)
        elif command == 'no_face':
            label, color = "NO FACE", (0, 0, 255)
        else:
            label, color = "FORWARD", (0, 255, 0)

        cv2.putText(frame, label, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.putText(frame, "SPACE=start/stop   DBL-BLINK=reverse   Q=quit",
                    (185, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    def _process_gaze_output(self, gaze_output):
        if gaze_output is None:
            return None

        # Detect double-blink toggle (blinked field flips on each double blink)
        blinked = gaze_output.get('blinked', False)
        if self._last_blinked_state is not None and blinked != self._last_blinked_state:
            self._reverse = not self._reverse
            print(f'[Gaze] Direction: {"REVERSE" if self._reverse else "FORWARD"}')
        self._last_blinked_state = blinked

        if not self.gaze_enabled:
            self.controller.stop()
            return 'paused'

        face_detected = gaze_output.get('face_detected', True)
        move_enabled = gaze_output.get('move_enabled', True)
        gaze_horizontal = float(gaze_output.get('combined_gaze_horizontal', 0.0))

        if self.face_required and not face_detected:
            self.controller.stop()
            return 'no_face'

        if not move_enabled:
            self.controller.stop()
            return 'disabled'

        command = self._map_gaze_to_command(gaze_horizontal)
        self._execute_command(command)
        return command

    def _map_gaze_to_command(self, horizontal):
        if horizontal < -self.horizontal_threshold:
            return 'left'
        if horizontal > self.horizontal_threshold:
            return 'right'
        # Straight ahead: default direction, flipped by double blink
        return 'backward' if self._reverse else 'forward'

    def _execute_command(self, command):
        if command == 'left':
            self.controller.move_left()
        elif command == 'right':
            self.controller.move_right()
        elif command == 'forward':
            self.controller.move_forward()
        elif command == 'backward':
            self.controller.move_backward()
        else:
            self.controller.stop()
