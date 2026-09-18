"""
Unified pipeline: builds any subset of the CereCe modules and runs one control
tick at a time.

The point of this class is that `main.py`, `tools/smoke_run.py` and the pytest
suite all drive the *same* wiring. Nothing here imports cv2, pandas or the gaze
tracker at module level — every real module is imported lazily inside its
factory — so the pipeline can be unit-tested with fakes in a bare Python
environment.

Module vocabulary
-----------------
User-facing alias -> config key:
    eeg     -> eeg_receiver
    gaze    -> gaze_receiver
    sonar   -> ultrasonic_receiver
    control -> wheelchair_controller

Degradation contract
--------------------
A module that is requested but cannot start is disabled for the run and
recorded in `Pipeline.degraded`; the remaining modules keep working. Only
`control` being unavailable changes the command path (it becomes dry-run).
"""

import time
from typing import Callable, Dict, List, Optional

EEG_WARMUP_SECONDS = 5.0
EEG_HEALTH_TIMEOUT = 5.0
SONAR_WARN_AFTER = 10.0

MODULE_ALIASES = {
    'eeg': 'eeg_receiver',
    'gaze': 'gaze_receiver',
    'sonar': 'ultrasonic_receiver',
    'control': 'wheelchair_controller',
}

# Canonical (config-key) module names, in display order.
MODULE_KEYS = (
    'eeg_receiver',
    'gaze_receiver',
    'ultrasonic_receiver',
    'wheelchair_controller',
)

MODULE_DEFAULTS = {
    'eeg_receiver': True,
    'gaze_receiver': True,
    'ultrasonic_receiver': False,
    'wheelchair_controller': True,
}

def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def parse_module_list(text) -> List[str]:
    """'eeg,gaze' (or ['eeg', 'gaze']) -> ['eeg_receiver', 'gaze_receiver']."""
    if not text:
        return []
    if isinstance(text, str):
        names = [p.strip() for p in text.replace(';', ',').split(',')]
    else:
        names = [p.strip() for p in text]
    keys = []
    for name in names:
        if not name:
            continue
        key = MODULE_ALIASES.get(name.lower(), name)
        if key not in MODULE_KEYS:
            raise ValueError(
                f"Unknown module '{name}'. Choose from: "
                + ', '.join(sorted(MODULE_ALIASES))
            )
        keys.append(key)
    return keys


def resolve_modules(config_modules=None, selection=None, enable=None,
                    disable=None) -> Dict[str, bool]:
    """
    Resolve the final module set.

    Precedence: built-in defaults < config['modules'] < `selection` < `enable`
    < `disable`.

    `selection` accepts either a mapping ({'eeg_receiver': True, ...}, applied
    as a per-key override) or an iterable of canonical keys ({'eeg_receiver'},
    which is exclusive: keys not listed are turned off).
    """
    resolved = dict(MODULE_DEFAULTS)
    for key, value in (config_modules or {}).items():
        if key in MODULE_KEYS:
            resolved[key] = bool(value)
    if selection:
        if isinstance(selection, dict):
            for key in MODULE_KEYS:
                if key in selection:
                    resolved[key] = bool(selection[key])
        else:
            chosen = set(selection)
            for key in MODULE_KEYS:
                resolved[key] = key in chosen
    for key in (enable or []):
        resolved[key] = True
    for key in (disable or []):
        resolved[key] = False
    return resolved


# ── Real module factories (lazy imports) ────────────────────────────────────
def _real_eeg_receiver(config):
    from modules.eeg_receiver import EEG_Receiver
    return EEG_Receiver(config)


def _real_mi_predictor(config):
    from modules.mi_predictor import MI_Predictor
    return MI_Predictor(config)


def _real_evidence_accumulator(config):
    from modules.evidence_accumulator import Evidence_Accumulator
    return Evidence_Accumulator(config)


def _real_gaze_receiver(config):
    from modules.gaze_receiver import Gaze_Receiver
    return Gaze_Receiver(config)


def _real_ultrasonic_receiver(config):
    from modules.ultrasonic_receiver import Ultrasonic_Receiver
    return Ultrasonic_Receiver(config)


def _real_wheelchair_controller(config):
    from modules.wheelchair_controller import Wheelchair_Controller
    return Wheelchair_Controller(config)


DEFAULT_FACTORIES: Dict[str, Callable[[dict], object]] = {
    'eeg_receiver': _real_eeg_receiver,
    'mi_predictor': _real_mi_predictor,
    'evidence_accumulator': _real_evidence_accumulator,
    'gaze_receiver': _real_gaze_receiver,
    'ultrasonic_receiver': _real_ultrasonic_receiver,
    'wheelchair_controller': _real_wheelchair_controller,
}


class Pipeline:
    """
    Owns the module instances and the drive/steering state machine.

    Usage:
        pipe = Pipeline(config)
        pipe.start()
        while running:
            status = pipe.step()
            ...render status...
        pipe.close()
    """

    def __init__(self, config: Optional[dict] = None, modules=None,
                 dry_run: bool = False, factories=None,
                 warmup_seconds: Optional[float] = None):
        self.config = config or {}
        self.dry_run = bool(dry_run)
        self.factories = dict(DEFAULT_FACTORIES)
        if factories:
            self.factories.update(factories)

        self.modules = resolve_modules(self.config.get('modules'), modules)
        self.degraded: Dict[str, str] = {}   # module key -> reason
        self.startup_notes: List[str] = []

        # Timing / policy
        self.command_keepalive = float(self.config.get('command_keepalive', 0.5))
        self.keyboard_hold = float(self.config.get('keyboard_override_hold', 0.25))
        self.health_timeout = float(self.config.get('eeg_health_timeout', EEG_HEALTH_TIMEOUT))
        if warmup_seconds is None:
            warmup_seconds = float(self.config.get('eeg_warmup_seconds', EEG_WARMUP_SECONDS))
        self.warmup_seconds = float(warmup_seconds)
        self.exit_on_source_end = as_bool(self.config.get('exit_on_source_end', False))
        self.show_camera_ui = as_bool(self.config.get('show_camera_ui', True))

        # Module instances
        self.receiver = None
        self.predictor = None
        self.accumulator = None
        self.gaze = None
        self.sonar = None
        self.controller = None

        # `eeg_requested` is captured before any degradation so that
        # `start_drive_enabled: auto` cannot arm the chair just because a
        # requested EEG failed to start.
        self.eeg_requested = self.modules['eeg_receiver']
        self.control_enabled = self.modules['wheelchair_controller']

        # Control state
        self.eeg_gate = False
        self.drive_enabled = False
        self.prev_mi_gate = 'inactive'
        self.reverse_gear = False
        self.prev_blink = False
        self.force_kb_steering = False
        self.keyboard_direction: Optional[str] = None
        self.keyboard_override_until = 0.0
        self.last_command: Optional[str] = None
        self.last_command_time = 0.0
        self.sonar_warned = False
        self.started_at: Optional[float] = None

        self._events: List[str] = []

        self._build()
        self._set_initial_drive()

    # ── Construction ─────────────────────────────────────────────────────
    def _build(self):
        if self.modules['eeg_receiver']:
            try:
                self.receiver = self.factories['eeg_receiver'](
                    self.config.get('receiver_params', {}))
                self.predictor = self.factories['mi_predictor'](
                    self.config.get('predictor_params', {}))
                self.accumulator = self.factories['evidence_accumulator'](
                    self.config.get('accumulator_params', {}))
            except Exception as e:
                self.receiver = self.predictor = self.accumulator = None
                self._degrade('eeg_receiver', f'init failed: {e}')

        if self.modules['gaze_receiver']:
            try:
                self.gaze = self.factories['gaze_receiver'](
                    self.config.get('gaze_params', {}))
            except Exception as e:
                self.gaze = None
                self._degrade('gaze_receiver', f'init failed: {e}')

        if self.modules['ultrasonic_receiver']:
            try:
                sonar = self.factories['ultrasonic_receiver'](
                    self.config.get('ultrasonic_params', {}))
                if getattr(sonar, 'available', True):
                    self.sonar = sonar
                else:
                    self._degrade('ultrasonic_receiver', 'bleak unavailable')
            except Exception as e:
                self.sonar = None
                self._degrade('ultrasonic_receiver', f'init failed: {e}')

        if self.modules['wheelchair_controller']:
            control_config = dict(self.config.get('control_params', {}))
            if self.dry_run:
                control_config['debug_mode'] = True
            try:
                self.controller = self.factories['wheelchair_controller'](control_config)
            except Exception as e:
                self.controller = None
                self._degrade('wheelchair_controller', f'init failed: {e}')

    def _degrade(self, key: str, reason: str):
        self.modules[key] = False
        self.degraded[key] = reason
        if key == 'wheelchair_controller':
            self.control_enabled = False
        self.startup_notes.append(f"{key} disabled — {reason}")

    def _set_initial_drive(self):
        start = self.config.get('start_drive_enabled', 'auto')
        if isinstance(start, str) and start.strip().lower() == 'auto':
            self.drive_enabled = not self.eeg_requested
        else:
            self.drive_enabled = as_bool(start)

    # ── Lifecycle ────────────────────────────────────────────────────────
    def start(self) -> dict:
        """Start the module threads, run the EEG warmup, return a summary."""
        if self.receiver is not None:
            self.receiver.start()
        if self.gaze is not None:
            self.gaze.start()
        if self.sonar is not None:
            self.sonar.start()

        if self.eeg_requested and self.receiver is not None:
            if self.warmup_seconds > 0:
                self.startup_notes.append(
                    f"waiting {self.warmup_seconds:g}s for EEG warmup")
                time.sleep(self.warmup_seconds)
            self.eeg_gate = bool(self.receiver.is_healthy(self.health_timeout))
            if self.eeg_gate:
                self.startup_notes.append("EEG healthy — MI gate is live")
            else:
                self.startup_notes.append(
                    "EEG not collecting data — MI gate unavailable; "
                    "steering continues without it")
        elif self.eeg_requested:
            self.startup_notes.append("EEG unavailable — MI gate unavailable")

        self.started_at = time.time()
        return self.summary()

    def close(self) -> List[str]:
        """Stop everything. Motors are stopped first and on every path."""
        events: List[str] = []
        if self.controller is not None:
            try:
                self.controller.stop()
                events.append("Motors stopped.")
            except Exception as e:
                events.append(f"Failed to stop motors: {e}")
        if self.gaze is not None:
            try:
                self.gaze.stop()
            except Exception as e:
                events.append(f"Gaze stop failed: {e}")
        if self.sonar is not None:
            try:
                self.sonar.stop()
            except Exception as e:
                events.append(f"Sonar stop failed: {e}")
        if self.receiver is not None:
            try:
                self.receiver.stop()
                self.receiver.join(timeout=2.0)
            except Exception as e:
                events.append(f"EEG stop failed: {e}")
        return events

    # ── Introspection ────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            'enabled': [k for k in MODULE_KEYS if self.modules[k]],
            'degraded': dict(self.degraded),
            'eeg_gate': self.eeg_gate,
            'drive_enabled': self.drive_enabled,
            'mode_label': self.mode_label(),
            'notes': list(self.startup_notes),
        }

    def mode_base(self) -> str:
        if self.force_kb_steering:
            return 'KB OVERRIDE'
        if self.eeg_gate and self.gaze is not None:
            return 'EEG+GAZE'
        if self.eeg_gate:
            return 'EEG+KB'
        if self.gaze is not None:
            return 'GAZE+KB'
        return 'KEYBOARD'

    def mode_label(self) -> str:
        return self.mode_base() + ('+SONAR' if self.sonar is not None else '')

    # ── Inputs ───────────────────────────────────────────────────────────
    def set_keyboard(self, direction: Optional[str], now: Optional[float] = None):
        """
        Report the currently held WASD direction, or None for "no key held".

        A direction is latched for `keyboard_override_hold` seconds so the
        override still works where cv2 does not emit auto-repeat events.
        """
        now = time.time() if now is None else now
        if direction is not None:
            self.keyboard_direction = direction
            self.keyboard_override_until = now + self.keyboard_hold
        elif now > self.keyboard_override_until:
            self.keyboard_direction = None

    def toggle_drive(self) -> str:
        """Spacebar handler. Returns 'gated', 'enabled' or 'disabled'."""
        if self.eeg_gate:
            return 'gated'
        self.drive_enabled = not self.drive_enabled
        return 'enabled' if self.drive_enabled else 'disabled'

    def toggle_keyboard_override(self) -> bool:
        self.force_kb_steering = not self.force_kb_steering
        return self.force_kb_steering

    def request_drive(self, enabled: bool):
        self.drive_enabled = bool(enabled)

    # ── Tick ─────────────────────────────────────────────────────────────
    def step(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        self._events = []

        status = {
            'mi_gate': None,
            'direction': 'stop',
            'steering_source': 'none',
            'drive_enabled': self.drive_enabled,
            'reverse_gear': self.reverse_gear,
            'obstacle': False,
            'command': None,
            'command_sent': False,
            'source_exhausted': False,
            'source_state': None,
            'sonar_distances': None,
            'sonar_healthy': None,
            'mode_label': self.mode_label(),
            'mode_base': self.mode_base(),
            'events': self._events,
        }

        # ── MI gate ──────────────────────────────────────────────────────
        if self.eeg_requested and self.receiver is not None:
            data = self.receiver.get_buffer_data()
            raw = (self.predictor.process_and_predict(data)
                   if len(data) > 0 else 'none')
            mi_gate = self.accumulator.update(raw)
            status['mi_gate'] = mi_gate

            if mi_gate == 'active' and self.prev_mi_gate == 'inactive':
                self.drive_enabled = not self.drive_enabled
                self._event(f"Drive {'ENABLED' if self.drive_enabled else 'DISABLED'} (MI gate)")
            self.prev_mi_gate = mi_gate

            if self.receiver.is_healthy(self.health_timeout):
                if not self.eeg_gate:
                    self.eeg_gate = True
                    self.drive_enabled = False
                    self.prev_mi_gate = 'inactive'
                    self._event("EEG signal detected — MI gate live "
                                "(drive disarmed; MI toggles it)")
            else:
                if self.eeg_gate:
                    self._event("EEG data loss — MI gate unavailable; "
                                "steering continues without it")
                self.eeg_gate = False
                self.prev_mi_gate = 'inactive'
        else:
            self.eeg_gate = False

        # Only a finite source (file replay) can be exhausted; a live headset
        # streams until stopped and never sets this.
        if self.receiver is not None:
            status['source_exhausted'] = bool(
                getattr(self.receiver, 'source_exhausted', False))
            source_state = getattr(self.receiver, 'state', None)
            status['source_state'] = getattr(source_state, 'value', source_state)

        # ── Steering: held key wins, then gaze, else stop ────────────────
        if self.force_kb_steering:
            direction = self.keyboard_direction or 'stop'
            steering_source = 'keyboard (override)'
        elif self.keyboard_direction is not None:
            direction = self.keyboard_direction
            steering_source = 'keyboard'
        elif self.gaze is not None:
            gaze_direction = self.gaze.get_direction()
            if gaze_direction is None:
                direction, steering_source = 'stop', 'gaze (warming up)'
            else:
                direction, steering_source = gaze_direction, 'gaze'
        else:
            direction, steering_source = 'stop', 'none'

        # ── Blink gear toggle (needs gaze, independent of steering) ──────
        if self.gaze is not None:
            blink = bool(self.gaze.get_blink())
            if blink and not self.prev_blink:
                self.reverse_gear = not self.reverse_gear
                self._event(f"Gear: {'REVERSE' if self.reverse_gear else 'FORWARD'} (blink)")
            self.prev_blink = blink

        if self.reverse_gear:
            if direction == 'forward':
                direction = 'backward'
            elif direction == 'backward':
                direction = 'forward'

        # ── Obstacle guard ───────────────────────────────────────────────
        if self.sonar is not None:
            obstacle = bool(self.sonar.is_obstacle())
            direction = self.sonar.filter_direction(direction, obstacle)
            healthy = bool(self.sonar.is_healthy())
            status['obstacle'] = obstacle
            status['sonar_healthy'] = healthy
            status['sonar_distances'] = self.sonar.get_distances()
            reference = self.started_at if self.started_at is not None else now
            if (not self.sonar_warned and not healthy
                    and (now - reference) > SONAR_WARN_AFTER):
                self.sonar_warned = True
                self._event("WARNING: ultrasonic sensor is not reporting — "
                            "obstacle guard inactive.")

        # ── Command (deduplicated + keepalive) ───────────────────────────
        command = direction if self.drive_enabled else 'stop'
        due = ((command != self.last_command)
               or (now - self.last_command_time) >= self.command_keepalive)
        if due:
            if self.controller is not None:
                self._dispatch(command)
            if command != self.last_command:
                self._event(f"Command: {command} (steering: {steering_source})")
            self.last_command = command
            self.last_command_time = now

        status['direction'] = direction
        status['steering_source'] = steering_source
        status['drive_enabled'] = self.drive_enabled
        status['reverse_gear'] = self.reverse_gear
        status['command'] = command
        status['command_sent'] = due
        status['mode_label'] = self.mode_label()
        status['mode_base'] = self.mode_base()
        return status

    # ── Internals ────────────────────────────────────────────────────────
    def _event(self, message: str):
        self._events.append(message)

    def _dispatch(self, command: str):
        if command == 'forward':
            self.controller.move_forward()
        elif command == 'backward':
            self.controller.move_backward()
        elif command == 'left':
            self.controller.move_left()
        elif command == 'right':
            self.controller.move_right()
        else:
            self.controller.stop()
