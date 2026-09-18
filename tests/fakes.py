"""
Fake module implementations for tests and the headless smoke runner.

Standard library only — no numpy, cv2, pandas, sklearn or bleak — so the
pipeline can be exercised anywhere. Each fake reads its script from the same
config sub-dict the real module would receive, which keeps tests declarative:

    config = {
        'modules': {'eeg_receiver': True, ...},
        'receiver_params': {'healthy': False},
        'gaze_params': {'directions': ['forward', 'left']},
        ...
    }
"""


class FakeEEGReceiver:
    """Stand-in for modules.eeg_receiver.EEG_Receiver."""

    def __init__(self, params=None):
        params = params or {}
        self.healthy = params.get('healthy', True)
        self.finished = bool(params.get('finished', False))
        self.finish_after = params.get('finish_after')       # polls before finishing
        self.samples = int(params.get('samples', 2000))
        self.started = False
        self.stopped = False
        self.polls = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def join(self, timeout=None):
        return None

    def is_healthy(self, timeout=5.0):
        return self.healthy

    def get_buffer_data(self):
        self.polls += 1
        if self.finish_after and self.polls >= int(self.finish_after):
            self.finished = True
        # Plain list: the pipeline only checks len(); no numpy needed.
        return [[0.0] * 5] * (self.samples if self.healthy else 0)


class FakePredictor:
    """Stand-in for MI_Predictor. `predictions` is cycled per call."""

    def __init__(self, params=None):
        params = params or {}
        self.predictions = list(params.get('predictions', ['none']))
        self.calls = 0

    def process_and_predict(self, data):
        if not self.predictions:
            return 'none'
        value = self.predictions[min(self.calls, len(self.predictions) - 1)]
        self.calls += 1
        return value


class FakeAccumulator:
    """Stand-in for Evidence_Accumulator. Active on the Nth call, then idle."""

    def __init__(self, params=None):
        params = params or {}
        self.activate_at = params.get('activate_at')
        self.sequence = list(params.get('sequence', []))
        self.calls = 0

    def update(self, raw_prediction):
        self.calls += 1
        if self.sequence:
            return self.sequence[min(self.calls - 1, len(self.sequence) - 1)]
        if self.activate_at and self.calls == int(self.activate_at):
            return 'active'
        return 'inactive'


class FakeGaze:
    """Stand-in for Gaze_Receiver.

    Params:
        directions (list): cycled per get_direction() call. Default ['stop'].
        blinks (list):     cycled per get_blink() call. Default [False].
        fail (bool):       raise on construction, as a dead camera would.
    """

    def __init__(self, params=None):
        params = params or {}
        if params.get('fail'):
            raise RuntimeError(params.get('fail_reason', 'fake camera failure'))
        self.directions = list(params.get('directions', ['stop']))
        self.blinks = list(params.get('blinks', [False]))
        self.started = False
        self.stopped = False
        self._d = 0
        self._b = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_direction(self):
        if not self.directions:
            return 'stop'
        value = self.directions[min(self._d, len(self.directions) - 1)]
        self._d += 1
        return value

    def get_blink(self):
        if not self.blinks:
            return False
        value = self.blinks[min(self._b, len(self.blinks) - 1)]
        self._b += 1
        return bool(value)

    def get_frame(self):
        return None


class FakeSonar:
    """Stand-in for Ultrasonic_Receiver.

    Params:
        available (bool):  False models a missing `bleak` install.
        healthy (bool):    False models "not reporting".
        obstacle (bool):   what is_obstacle() returns while healthy.
        treat_unhealthy_as_obstacle (bool): mirror of the real fail-safe option.
        block_directions (list)
        distances (tuple)
    """

    def __init__(self, params=None):
        params = params or {}
        self.available = params.get('available', True)
        self.healthy = params.get('healthy', True)
        self.obstacle = bool(params.get('obstacle', False))
        self.treat_unhealthy_as_obstacle = bool(
            params.get('treat_unhealthy_as_obstacle', False))
        self.block_directions = set(params.get('block_directions', ['forward']))
        self.distances = tuple(params.get('distances', (120.0, 120.0)))
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def get_distances(self):
        return self.distances if self.healthy else None

    def is_healthy(self):
        return self.healthy

    def is_obstacle(self):
        if not self.healthy:
            return self.treat_unhealthy_as_obstacle
        return self.obstacle

    def filter_direction(self, direction, obstacle=None):
        if obstacle is None:
            obstacle = self.is_obstacle()
        if obstacle and direction in self.block_directions:
            return 'stop'
        return direction


class RecordingController:
    """Stand-in for Wheelchair_Controller that records instead of driving."""

    instances = []

    def __init__(self, config=None):
        self.config = config or {}
        self.debug_mode = self.config.get('debug_mode', True)
        self.commands = []
        RecordingController.instances.append(self)

    def _record(self, command):
        self.commands.append(command)

    def move_forward(self):
        self._record('forward')

    def move_backward(self):
        self._record('backward')

    def move_left(self):
        self._record('left')

    def move_right(self):
        self._record('right')

    def stop(self):
        self._record('stop')

    @classmethod
    def reset(cls):
        cls.instances = []

    @classmethod
    def last(cls):
        return cls.instances[-1] if cls.instances else None


def fake_factories(**overrides):
    """Factory dict for Pipeline(factories=...)."""
    factories = {
        'eeg_receiver': FakeEEGReceiver,
        'mi_predictor': FakePredictor,
        'evidence_accumulator': FakeAccumulator,
        'gaze_receiver': FakeGaze,
        'ultrasonic_receiver': FakeSonar,
        'wheelchair_controller': RecordingController,
    }
    factories.update(overrides)
    return factories


def base_config(**overrides):
    """A complete, fake-friendly pipeline config."""
    config = {
        'loop_interval': 0.005,
        'show_camera_ui': False,
        'headless': True,
        'run_duration': 0,
        'exit_on_source_end': False,
        'command_keepalive': 10.0,
        'keyboard_override_hold': 0.06,
        'start_drive_enabled': 'auto',
        'eeg_warmup_seconds': 0,
        'modules': {
            'eeg_receiver': False,
            'gaze_receiver': False,
            'ultrasonic_receiver': False,
            'wheelchair_controller': True,
        },
        'receiver_params': {},
        'predictor_params': {},
        'accumulator_params': {},
        'control_params': {},
        'gaze_params': {},
        'ultrasonic_params': {},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config[key])
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    return config
