"""
Behavioural tests for the unified pipeline's module subsets and gating.

Everything here uses tests/fakes.py, so no camera, EEG, BLE sensor or
wheelchair is needed.
"""

import pytest

from modules.pipeline import (
    Pipeline,
    parse_module_list,
    resolve_modules,
)
from tests.fakes import RecordingController, base_config, fake_factories

ON = True
OFF = False


def make(config=None, modules=None, **kwargs):
    RecordingController.reset()
    return Pipeline(
        config or base_config(),
        modules=modules,
        factories=fake_factories(),
        warmup_seconds=0,
        **kwargs,
    )


def modules(eeg=OFF, gaze=OFF, sonar=OFF, control=ON):
    return {
        'eeg_receiver': eeg,
        'gaze_receiver': gaze,
        'ultrasonic_receiver': sonar,
        'wheelchair_controller': control,
    }


# ── Steering sources ────────────────────────────────────────────────────────

def test_gaze_only_steers_with_gaze():
    config = base_config(modules=modules(gaze=ON),
                         gaze_params={'directions': ['forward']})
    pipe = make(config)
    pipe.start()
    status = pipe.step(now=0.0)
    assert status['steering_source'] == 'gaze'
    assert status['command'] == 'forward'
    # No EEG was requested, so `auto` arms the drive immediately.
    assert status['drive_enabled'] is True
    pipe.close()


def test_held_key_overrides_gaze_then_release_returns_to_gaze():
    config = base_config(modules=modules(gaze=ON),
                         gaze_params={'directions': ['left']},
                         keyboard_override_hold=0.06)
    pipe = make(config)
    pipe.start()

    assert pipe.step(now=0.0)['command'] == 'left'

    pipe.set_keyboard('right', now=0.01)
    override = pipe.step(now=0.02)
    assert override['steering_source'] == 'keyboard'
    assert override['command'] == 'right'

    # Still inside the latch window: the key is treated as held.
    pipe.set_keyboard(None, now=0.03)
    assert pipe.step(now=0.04)['command'] == 'right'

    # Past the latch window: gaze takes over again.
    pipe.set_keyboard(None, now=0.10)
    assert pipe.step(now=0.11)['command'] == 'left'
    pipe.close()


def test_keyboard_only_when_gaze_is_off():
    pipe = make(base_config(modules=modules()))
    pipe.start()
    assert pipe.step(now=0.0)['command'] == 'stop'

    pipe.set_keyboard('forward', now=0.01)
    status = pipe.step(now=0.02)
    assert status['steering_source'] == 'keyboard'
    assert status['command'] == 'forward'
    pipe.close()


def test_sticky_keyboard_override_ignores_gaze():
    config = base_config(modules=modules(gaze=ON),
                         gaze_params={'directions': ['left']})
    pipe = make(config)
    pipe.start()
    assert pipe.step(now=0.0)['command'] == 'left'

    assert pipe.toggle_keyboard_override() is True
    status = pipe.step(now=0.1)
    assert status['steering_source'] == 'keyboard (override)'
    assert status['command'] == 'stop'   # override with no key held == stop
    pipe.close()


# ── MI gate ─────────────────────────────────────────────────────────────────

def test_mi_gate_toggles_drive():
    config = base_config(modules=modules(eeg=ON),
                         accumulator_params={'activate_at': 2})
    pipe = make(config)
    pipe.start()
    assert pipe.eeg_gate is True
    assert pipe.drive_enabled is False   # EEG requested: start disarmed

    assert pipe.step(now=0.0)['mi_gate'] == 'inactive'
    assert pipe.step(now=0.1)['drive_enabled'] is True
    assert pipe.step(now=0.2)['drive_enabled'] is True
    pipe.close()


def test_spacebar_is_gated_when_mi_gate_is_live():
    pipe = make(base_config(modules=modules(eeg=ON)))
    pipe.start()
    assert pipe.eeg_gate is True
    assert pipe.toggle_drive() == 'gated'
    assert pipe.drive_enabled is False
    pipe.close()


def test_spacebar_toggles_drive_when_no_gate_exists():
    pipe = make(base_config(modules=modules(gaze=ON)))
    pipe.start()
    assert pipe.eeg_gate is False
    assert pipe.drive_enabled is True
    assert pipe.toggle_drive() == 'disabled'
    assert pipe.drive_enabled is False
    pipe.close()


def test_eeg_requested_but_unhealthy_never_auto_arms():
    config = base_config(modules=modules(eeg=ON),
                         receiver_params={'healthy': False})
    pipe = make(config)
    pipe.start()
    assert pipe.eeg_gate is False
    assert pipe.drive_enabled is False

    pipe.set_keyboard('forward', now=0.0)
    assert pipe.step(now=0.0)['command'] == 'stop'
    pipe.close()


def test_source_exhausted_only_for_finite_sources():
    config = base_config(modules=modules(eeg=ON),
                         receiver_params={'finish_after': 1},
                         exit_on_source_end=True)
    pipe = make(config)
    pipe.start()
    status = pipe.step(now=0.0)
    assert status['source_exhausted'] is True
    assert status['source_state'] == 'exhausted'
    assert pipe.exit_on_source_end is True
    pipe.close()


def test_continuous_source_is_never_exhausted():
    """A live headset streams until stopped; it must not look like a replay."""
    config = base_config(modules=modules(eeg=ON), exit_on_source_end=True)
    pipe = make(config)
    pipe.start()
    for i in range(5):
        status = pipe.step(now=i * 0.1)
        assert status['source_exhausted'] is False
        assert status['source_state'] == 'streaming'
    pipe.close()


# ── Graceful degradation ────────────────────────────────────────────────────

def _boom(_config):
    raise RuntimeError('no device')


def test_eeg_failure_degrades_without_disarming_other_modules():
    config = base_config(modules=modules(eeg=ON, gaze=ON),
                         gaze_params={'directions': ['forward']})
    pipe = Pipeline(config, modules=modules(eeg=ON, gaze=ON),
                    factories=fake_factories(eeg_receiver=_boom),
                    warmup_seconds=0)
    pipe.start()
    assert 'eeg_receiver' in pipe.degraded
    assert pipe.eeg_gate is False
    # EEG was requested, so a dead EEG must not arm the drive.
    assert pipe.drive_enabled is False

    status = pipe.step(now=0.0)
    assert status['steering_source'] == 'gaze'
    assert status['command'] == 'stop'    # drive is disarmed, steering still resolves
    pipe.close()


def test_gaze_failure_degrades_to_keyboard():
    config = base_config(modules=modules(gaze=ON),
                         gaze_params={'fail': True})
    pipe = make(config)
    pipe.start()
    assert 'gaze_receiver' in pipe.degraded
    assert pipe.gaze is None

    pipe.set_keyboard('left', now=0.0)
    status = pipe.step(now=0.0)
    assert status['steering_source'] == 'keyboard'
    assert status['command'] == 'left'
    pipe.close()


def test_sonar_unavailable_degrades_and_stops_blocking():
    config = base_config(modules=modules(gaze=ON, sonar=ON),
                         gaze_params={'directions': ['forward']},
                         ultrasonic_params={'available': False, 'obstacle': True})
    pipe = make(config)
    pipe.start()
    assert 'ultrasonic_receiver' in pipe.degraded
    assert pipe.sonar is None
    assert pipe.step(now=0.0)['command'] == 'forward'
    pipe.close()


def test_sonar_blocks_only_configured_directions():
    config = base_config(modules=modules(gaze=ON, sonar=ON),
                         gaze_params={'directions': ['left', 'forward']},
                         ultrasonic_params={'obstacle': True,
                                            'block_directions': ['forward']})
    pipe = make(config)
    pipe.start()
    assert pipe.step(now=0.0)['command'] == 'left'
    blocked = pipe.step(now=0.1)
    assert blocked['obstacle'] is True
    assert blocked['direction'] == 'stop'
    pipe.close()


# ── Command path ────────────────────────────────────────────────────────────

def test_command_is_deduplicated_and_keepalive_resends():
    config = base_config(modules=modules(gaze=ON),
                         gaze_params={'directions': ['forward']},
                         command_keepalive=0.5)
    pipe = make(config)
    pipe.start()

    assert pipe.step(now=0.0)['command_sent'] is True
    assert pipe.step(now=0.1)['command_sent'] is False
    assert pipe.step(now=0.6)['command_sent'] is True
    assert RecordingController.last().commands == ['forward', 'forward']
    pipe.close()


def test_blink_toggles_reverse_gear():
    config = base_config(modules=modules(gaze=ON),
                         gaze_params={'directions': ['forward'],
                                      'blinks': [False, True]})
    pipe = make(config)
    pipe.start()

    assert pipe.step(now=0.0)['direction'] == 'forward'
    flipped = pipe.step(now=0.1)
    assert flipped['reverse_gear'] is True
    assert flipped['direction'] == 'backward'
    pipe.close()


def test_close_always_stops_motors():
    config = base_config(modules=modules(gaze=ON),
                         gaze_params={'directions': ['forward']})
    pipe = make(config)
    pipe.start()
    pipe.step(now=0.0)
    assert pipe.drive_enabled is True

    events = pipe.close()
    assert 'Motors stopped.' in events
    assert RecordingController.last().commands[-1] == 'stop'


def test_no_controller_when_control_disabled():
    pipe = make(base_config(modules=modules(gaze=ON, control=OFF)))
    pipe.start()
    status = pipe.step(now=0.0)
    assert pipe.controller is None
    assert pipe.control_enabled is False
    assert status['command'] is not None   # still computed for dry-run display
    assert RecordingController.instances == []
    pipe.close()


def test_dry_run_forces_controller_debug_mode():
    config = base_config(modules=modules(), control_params={'debug_mode': False})
    RecordingController.reset()
    pipe = Pipeline(config, modules=modules(), factories=fake_factories(),
                    dry_run=True, warmup_seconds=0)
    assert pipe.controller.debug_mode is True
    pipe.close()


def test_mode_label_tracks_active_modules():
    config = base_config(modules=modules(gaze=ON, sonar=ON))
    pipe = make(config)
    pipe.start()
    assert pipe.step(now=0.0)['mode_label'] == 'GAZE+KB+SONAR'

    eeg_config = base_config(modules=modules(eeg=ON, gaze=ON))
    pipe2 = make(eeg_config)
    pipe2.start()
    assert pipe2.step(now=0.0)['mode_label'] == 'EEG+GAZE'
    pipe2.close()
    pipe.close()


# ── Helpers ─────────────────────────────────────────────────────────────────

def test_parse_module_list_accepts_aliases():
    assert parse_module_list('eeg,control') == ['eeg_receiver', 'wheelchair_controller']
    assert parse_module_list(['gaze', 'sonar']) == ['gaze_receiver', 'ultrasonic_receiver']
    assert parse_module_list('eeg_receiver') == ['eeg_receiver']
    assert parse_module_list('') == []


def test_parse_module_list_rejects_unknown_names():
    with pytest.raises(ValueError):
        parse_module_list('eeg,bogus')


def test_resolve_modules_precedence():
    assert resolve_modules({'eeg_receiver': True})['eeg_receiver'] is True
    # selection replaces the config set entirely
    resolved = resolve_modules({'eeg_receiver': True}, selection=['gaze_receiver'])
    assert resolved['gaze_receiver'] is True
    assert resolved['eeg_receiver'] is False
    # enable/disable win over both
    resolved = resolve_modules({}, selection=['gaze_receiver'],
                               enable=['ultrasonic_receiver'],
                               disable=['gaze_receiver'])
    assert resolved['ultrasonic_receiver'] is True
    assert resolved['gaze_receiver'] is False


def test_resolve_modules_accepts_a_mapping_selection():
    # Regression: a mapping must be honoured per key. `set(dict)` yields every
    # key, which previously turned every module on regardless of its value.
    resolved = resolve_modules({}, selection={'gaze_receiver': True,
                                              'eeg_receiver': False})
    assert resolved['gaze_receiver'] is True
    assert resolved['eeg_receiver'] is False
    # Keys absent from the mapping keep their default rather than being forced on.
    assert resolved['ultrasonic_receiver'] is False


def test_pipeline_builds_only_the_selected_modules():
    config = base_config(gaze_params={'directions': ['forward']})
    pipe = Pipeline(config, modules=modules(gaze=ON, control=OFF),
                    factories=fake_factories(), warmup_seconds=0)
    pipe.start()
    assert pipe.gaze is not None
    assert pipe.receiver is None
    assert pipe.sonar is None
    assert pipe.controller is None        # control was not part of the selection
    assert pipe.step(now=0.0)['mode_label'] == 'GAZE+KB'
    pipe.close()


def test_pipeline_accepts_an_iterable_selection():
    pipe = Pipeline(base_config(), modules={'ultrasonic_receiver'},
                    factories=fake_factories(), warmup_seconds=0)
    pipe.start()
    assert pipe.sonar is not None
    assert pipe.gaze is None
    assert pipe.receiver is None
    assert pipe.controller is None
    pipe.close()
