"""
Unit tests for the individual modules, independent of the control loop.

Tests that need the project's heavy dependencies (pandas, scipy) skip
themselves when those are unavailable, and anything needing real hardware is
marked `hardware` and deselected by default (see pytest.ini).
"""

import contextlib
import io
import time
import types

import pytest

from modules.evidence_accumulator import Evidence_Accumulator
from modules.ultrasonic_receiver import Ultrasonic_Receiver


# ── Ultrasonic obstacle sensor ──────────────────────────────────────────────

def _sonar(**overrides):
    config = {'device_name': 'TestBLE'}
    config.update(overrides)
    return Ultrasonic_Receiver(config)


def test_notify_parses_lhs_rhs_payload():
    rx = _sonar()
    rx._on_notify(None, bytearray(b'12.34,56.78'))
    assert rx.get_distances() == (12.34, 56.78)
    assert rx.last_data_time is not None


def test_malformed_packet_keeps_last_reading():
    rx = _sonar()
    rx._on_notify(None, bytearray(b'12.34,56.78'))
    with contextlib.redirect_stdout(io.StringIO()):
        rx._on_notify(None, bytearray(b'not-a-packet'))
    assert rx.get_distances() == (12.34, 56.78)


def test_obstacle_threshold_uses_either_sensor():
    rx = _sonar(obstacle_distance=50)
    rx.connected = True
    rx._lhs, rx._rhs = 49.0, 200.0
    assert rx.is_obstacle() is True
    rx._lhs, rx._rhs = 51.0, 200.0
    assert rx.is_obstacle() is False
    rx._lhs, rx._rhs = 200.0, 49.0
    assert rx.is_obstacle() is True


def test_unhealthy_sensor_warns_once_and_fails_open():
    rx = _sonar()
    rx.connected = False
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        assert rx.is_obstacle() is False
        assert rx.is_obstacle() is False
    assert captured.getvalue().count('WARNING') == 1


def test_unhealthy_sensor_can_fail_safe():
    rx = _sonar(treat_unhealthy_as_obstacle=True)
    rx.connected = False
    assert rx.is_obstacle() is True


def test_filter_direction_respects_block_list_and_precomputed_obstacle():
    rx = _sonar(block_directions=['forward'])
    assert rx.filter_direction('forward', obstacle=True) == 'stop'
    assert rx.filter_direction('left', obstacle=True) == 'left'
    assert rx.filter_direction('forward', obstacle=False) == 'forward'


def test_unavailable_sonar_is_marked_unavailable():
    rx = _sonar()
    # bleak is not installed in a bare test environment; either way the flag
    # must be a real bool so the pipeline can degrade cleanly.
    assert isinstance(rx.available, bool)


# ── Evidence accumulator ────────────────────────────────────────────────────

def test_accumulator_requires_threshold_then_holds_with_hysteresis():
    acc = Evidence_Accumulator({
        'threshold': 3.0, 'decay': 0.2, 'build_rate': 1.0,
        'max_evidence': 5.0, 'hysteresis_factor': 0.5,
    })
    for _ in range(3):
        assert acc.update('inactive') == 'inactive'

    outputs = [acc.update('active') for _ in range(4)]
    assert outputs[-1] == 'active'
    # A single dropped frame must not drop the state (hysteresis).
    assert acc.update('inactive') == 'active'


def test_accumulator_normalises_unknown_predictions_to_inactive():
    acc = Evidence_Accumulator({})
    assert acc.update('none') == 'inactive'
    assert acc.update('something-weird') == 'inactive'


# ── EEG receiver ────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    pd = pytest.importorskip('pandas')
    pd.DataFrame({
        'TimeStamp': [r[0] for r in rows],
        'Fcz': [r[1] for r in rows],
        'C3': [r[2] for r in rows],
        'Cz': [r[3] for r in rows],
        'C4': [r[4] for r in rows],
    }).to_csv(path, index=False)
    return path


def _file_receiver(path):
    from modules.eeg_receiver import EEG_Receiver
    return EEG_Receiver({'input_mode': 'file', 'data_path': str(path),
                         'buffer_size': 50})


def _source_state():
    pytest.importorskip('pandas')
    from modules.eeg_receiver import SourceState
    return SourceState


def test_file_replay_reaches_exhausted_and_fills_the_buffer(tmp_path):
    SourceState = _source_state()
    path = _write_csv(tmp_path / 'tiny.csv',
                      [(0.0, 1, 2, 3, 4), (0.001, 5, 6, 7, 8), (0.002, 9, 10, 11, 12)])
    rx = _file_receiver(path)
    try:
        # Constructed but never started: nothing has been asked of it yet.
        assert rx.state is SourceState.IDLE
        assert rx.source_exhausted is False

        rx.start()
        rx.join(timeout=5.0)

        assert rx.state is SourceState.EXHAUSTED
        assert rx.source_exhausted is True
        data = rx.get_buffer_data()
        assert data.shape == (3, 5)          # timestamp + 4 channels
        assert data[0][0] == pytest.approx(0.0)
        assert data[-1][0] == pytest.approx(0.002)
    finally:
        rx.stop()


def test_stopped_mid_replay_is_not_exhausted(tmp_path):
    """stop() during a replay must not look like a completed recording."""
    SourceState = _source_state()
    # Spaced timestamps so the replay is still running when stop() lands.
    rows = [(i * 0.05, 1, 2, 3, 4) for i in range(40)]
    path = _write_csv(tmp_path / 'longer.csv', rows)
    rx = _file_receiver(path)
    rx.start()
    rx.stop()
    rx.join(timeout=5.0)
    assert rx.state is SourceState.STOPPED
    assert rx.source_exhausted is False
    assert len(rx.get_buffer_data()) < len(rows)


def test_stop_before_thread_starts_is_honoured(tmp_path):
    """A stop() that lands before run() must not be undone by the thread."""
    SourceState = _source_state()
    rows = [(i * 0.05, 1, 2, 3, 4) for i in range(40)]
    path = _write_csv(tmp_path / 'longer.csv', rows)
    rx = _file_receiver(path)
    rx._stopped = True          # simulate stop() winning the race
    rx.start()
    rx.join(timeout=5.0)
    assert rx.running is False
    assert rx.source_exhausted is False
    assert rx.state is SourceState.STOPPED


def test_continuous_source_streams_and_is_never_exhausted(tmp_path):
    """
    The real-headset case: data keeps arriving, so the source is STREAMING and
    never EXHAUSTED. There is no "finished" for live EEG.
    """
    np = pytest.importorskip('numpy')
    SourceState = _source_state()
    path = _write_csv(tmp_path / 'tiny.csv', [(0.0, 1, 2, 3, 4)])
    rx = _file_receiver(path)
    try:
        packet = types.SimpleNamespace(
            get_data=lambda: (2.5, np.array([[1.0], [3.0], [5.0], [7.0]])))
        for _ in range(3):
            rx.update_buffer(packet)
            assert rx.state is SourceState.STREAMING
            assert rx.source_exhausted is False
        assert rx.is_healthy() is True
    finally:
        rx.stop()


def test_stale_when_data_stops_arriving(tmp_path):
    SourceState = _source_state()
    path = _write_csv(tmp_path / 'tiny.csv', [(0.0, 1, 2, 3, 4)])
    rx = _file_receiver(path)
    try:
        rx.last_data_time = time.time() - 60      # last packet a minute ago
        assert rx.state is SourceState.STALE
        assert rx.is_healthy() is False
        assert rx.is_healthy(timeout=120) is True  # timeout is parameterisable
    finally:
        rx.stop()


def test_health_is_false_until_data_arrives(tmp_path):
    SourceState = _source_state()
    path = _write_csv(tmp_path / 'tiny.csv', [(0.0, 1, 2, 3, 4)])
    rx = _file_receiver(path)
    try:
        # File mode marks itself connected, but no packet has arrived yet.
        assert rx.last_data_time is None
        assert rx.is_healthy() is False
        assert rx.state is SourceState.IDLE
    finally:
        rx.stop()


def test_update_buffer_accepts_explorepy_packet_shape(tmp_path):
    np = pytest.importorskip('numpy')
    path = _write_csv(tmp_path / 'tiny.csv', [(0.0, 1, 2, 3, 4)])
    rx = _file_receiver(path)
    try:
        packet = types.SimpleNamespace(
            get_data=lambda: (2.5, np.array([[1.0, 2.0],
                                             [3.0, 4.0],
                                             [5.0, 6.0],
                                             [7.0, 8.0]])))
        rx.update_buffer(packet)
        data = rx.get_buffer_data()
        assert data.shape == (2, 5)
        assert data[0][0] == pytest.approx(2.5)
        assert list(data[1][1:]) == [2.0, 4.0, 6.0, 8.0]
        assert rx.is_healthy() is True
    finally:
        rx.stop()


# ── MI predictor ────────────────────────────────────────────────────────────

def _predictor(model_path='/nonexistent/model.sav'):
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    from modules.mi_predictor import MI_Predictor
    return MI_Predictor({
        'sf': 250, 'n_ch': 4, 'channels': ['Fcz', 'C3', 'Cz', 'C4'],
        'low_freq': 7, 'high_freq': 30, 'signal_len': 5,
        'model_path': model_path,
    })


def test_feature_vector_matches_training_layout():
    np = pytest.importorskip('numpy')
    predictor = _predictor()
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((4, 1250))

    features = predictor._get_features(epoch)
    assert len(features) == 69          # 3 channels after Cz removal x 23 bins

    # Same slicing the training pipeline used: 7..29 Hz inclusive, 30 exclusive.
    _, freqs = predictor._psd_epoch(epoch[:3], predictor.sf,
                                    predictor.low_freq, predictor.high_freq)
    assert len(freqs) == 23
    assert freqs[0] == 7 and freqs[-1] == 29


def test_missing_model_is_reported_explicitly(capsys):
    predictor = _predictor()
    assert predictor.model is None
    assert 'Model load failed' in capsys.readouterr().out


def test_insufficient_data_returns_none():
    np = pytest.importorskip('numpy')
    predictor = _predictor()
    assert predictor.process_and_predict(np.zeros((100, 4))) == 'none'


def test_feature_mismatch_is_rejected(capsys):
    np = pytest.importorskip('numpy')
    predictor = _predictor()

    class WrongModel:
        n_features_in_ = 5

        def predict(self, X):        # pragma: no cover - must not be reached
            raise AssertionError('predict() should not be called on a mismatch')

    predictor.model = WrongModel()
    assert predictor.process_and_predict(np.zeros((1250, 4))) == 'none'
    assert 'Feature mismatch' in capsys.readouterr().out


# ── Real hardware (deselected by default) ───────────────────────────────────

@pytest.mark.hardware
def test_real_wheelchair_controller_constructs():  # pragma: no cover
    from modules.wheelchair_controller import Wheelchair_Controller
    controller = Wheelchair_Controller({'debug_mode': True})
    controller.stop()
