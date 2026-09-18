"""
CereCe MI + Gaze wheelchair controller — thin runner.

All module wiring and control logic lives in modules/pipeline.py. This file
only parses options, drives the loop, renders the camera overlay and reads the
keyboard, so any subset of modules can be run and tested:

    python main.py                                   # everything, per config.yaml
    python main.py --modules gaze                    # gaze + controller only
    python main.py --modules eeg,gaze --no-control --dry-run
    python main.py --modules sonar --enable gaze --dry-run
    python main.py --list-modules
    python main.py --headless --duration 30 --source file
"""

import argparse
import time

import cv2
import numpy as np

from utils.load_config import load_config
from modules.pipeline import (
    MODULE_ALIASES,
    Pipeline,
    as_bool,
    parse_module_list,
    resolve_modules,
)

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


def _print_startup(pipeline, headless, show_camera_ui, selection_overridden):
    print("[Main] Modules: " + ', '.join(
        f"{k}={'on' if pipeline.modules[k] else 'off'}" for k in pipeline.modules))
    if pipeline.degraded:
        for key, reason in pipeline.degraded.items():
            print(f"[Main] DEGRADED: {key} — {reason}")
    for note in pipeline.startup_notes:
        print(f"[Main] {note}")
    if selection_overridden:
        print("[Main] (module set overridden from the command line)")
    print(f"[Main] Drive starts {'ENABLED' if pipeline.drive_enabled else 'DISABLED'} "
          f"(start_drive_enabled={pipeline.config.get('start_drive_enabled', 'auto')}).")
    if pipeline.controller is None:
        print("[Main] wheelchair_controller inactive — DRY RUN (no motor commands).")
    elif pipeline.dry_run:
        print("[Main] --dry-run: controller built in debug mode (no motor commands).")
    elif pipeline.config.get('control_params', {}).get('debug_mode'):
        print("[Main] control_params.debug_mode is true — no motor commands.")
    else:
        print("[Main] *** LIVE MOTOR CONTROL ENABLED ***")
    if headless:
        print("[Main] Headless: no window and no keyboard input. Ctrl+C to stop.")
    else:
        keys = "WASD=steer  "
        if not pipeline.eeg_gate:
            keys += "SPACE=drive  "
        if pipeline.gaze is not None:
            keys += "G=keyboard override  "
        keys += "Q=quit"
        if pipeline.eeg_gate:
            keys += "  (drive is toggled by the MI gate)"
        print(f"[Main] Keys: {keys}")
        if not show_camera_ui:
            print("[Main] show_camera_ui is false — overlay-only panel keeps the keys alive.")


def _render(pipeline, status, show_camera_ui):
    frame = None
    if show_camera_ui and pipeline.gaze is not None:
        frame = pipeline.gaze.get_frame()
    if frame is None:
        frame = np.zeros((200, 640, 3), dtype=np.uint8)
        if show_camera_ui:
            cv2.putText(frame, 'Camera warming up...', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    label = 'HEADLESS' if status is None else status['mode_label']
    base = 'HEADLESS' if status is None else status['mode_base']
    color_mode = _MODE_COLORS.get(base, (200, 200, 200))
    direction = 'stop' if status is None else status['direction']
    drive_enabled = bool(pipeline.drive_enabled if status is None else status['drive_enabled'])
    reverse_gear = bool(pipeline.reverse_gear if status is None else status['reverse_gear'])

    cv2.putText(frame, f'Mode: {label}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_mode, 2, cv2.LINE_AA)
    cv2.putText(frame, f'Direction: {direction}', (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    drive_label = "DRIVE: ON" if drive_enabled else "DRIVE: OFF"
    drive_color = (0, 255, 0) if drive_enabled else (0, 0, 255)
    gear_label = "GEAR: REV" if reverse_gear else "GEAR: FWD"
    gear_color = (0, 165, 255) if reverse_gear else (255, 255, 255)
    cv2.putText(frame, drive_label, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, drive_color, 2, cv2.LINE_AA)
    cv2.putText(frame, gear_label, (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gear_color, 2, cv2.LINE_AA)

    if pipeline.sonar is not None and status is not None:
        dist = status['sonar_distances']
        if not status['sonar_healthy'] or dist is None:
            sonar_label, sonar_color = "SONAR: --", (128, 128, 128)
        elif status['obstacle']:
            sonar_label = f"OBSTACLE  L={dist[0]:.0f} R={dist[1]:.0f}cm"
            sonar_color = (0, 0, 255)
        else:
            sonar_label = f"SONAR: L={dist[0]:.0f} R={dist[1]:.0f}cm"
            sonar_color = (0, 255, 0)
        cv2.putText(frame, sonar_label, (frame.shape[1] - 330, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sonar_color, 2, cv2.LINE_AA)

    hints = "WASD=steer  Q=quit"
    if not pipeline.eeg_gate:
        hints = "SPACE=drive  " + hints
    if pipeline.gaze is not None:
        hints += "  G=keyboard override"
    cv2.putText(frame, hints, (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.imshow('Camera UI', frame)


def _read_keys(pipeline):
    """Returns True to keep running, False to quit."""
    key = cv2.waitKey(1) & 0xFF
    now = time.time()
    if key == ord('q'):
        return False
    try:
        if cv2.getWindowProperty('Camera UI', cv2.WND_PROP_VISIBLE) < 1:
            return False
    except cv2.error:
        pass

    direction_key = _KEY_MAP.get(key)
    if direction_key is not None:
        pipeline.set_keyboard(direction_key, now)
    elif key == 0xFF:
        pipeline.set_keyboard(None, now)

    if key == ord(' '):
        result = pipeline.toggle_drive()
        if result == 'gated':
            print("[Main] Drive is toggled by the MI gate; spacebar ignored.")
        else:
            print(f"[Main] Drive {result.upper()} (spacebar)")
    if key == ord('g'):
        on = pipeline.toggle_keyboard_override()
        print("[Main] Steering: "
              f"{'KEYBOARD OVERRIDE' if on else 'AUTO (held key > gaze)'} (g key)")
    return True


def _list_modules(config):
    resolved = resolve_modules(config.get('modules'))
    print("Available modules (alias -> config key):")
    for alias, key in MODULE_ALIASES.items():
        state = 'on' if resolved[key] else 'off'
        print(f"  {alias:<8} -> {key:<22} [{state} in this config]")
    print("\nExamples:")
    print("  python main.py --modules eeg,gaze")
    print("  python main.py --modules gaze --dry-run")
    print("  python main.py --modules control --dry-run --headless --duration 10")
    print("  python main.py --enable sonar --disable eeg --dry-run")


def start_MI_Tracking(config_path='config.yaml', modules=None, enable=None,
                      disable=None, dry_run=False, headless=None,
                      duration=None, source=None, warmup=None):
    config = load_config(config_path)

    # ── CLI overrides ─────────────────────────────────────────────────────
    if source:
        config.setdefault('receiver_params', {})['input_mode'] = source
    if headless:
        config['headless'] = True
    if duration is not None:
        config['run_duration'] = duration

    headless = as_bool(config.get('headless', False))
    show_camera_ui = as_bool(config.get('show_camera_ui', True))
    loop_interval = float(config.get('loop_interval', 0.05))
    run_duration = float(config.get('run_duration', 0) or 0)

    selection = resolve_modules(config.get('modules'), modules, enable, disable)

    pipeline = Pipeline(config, modules=selection, dry_run=dry_run,
                        warmup_seconds=warmup)
    pipeline.start()
    _print_startup(pipeline, headless, show_camera_ui,
                   selection_overridden=bool(modules or enable or disable))
    print("[Main] Control loop started.")

    start_time = time.time()
    try:
        while True:
            tick = time.time()

            status = pipeline.step(tick)
            tag = '[Main][DRY RUN]' if pipeline.controller is None else '[Main]'
            for event in status['events']:
                print(f"{tag} {event}")

            if not headless:
                _render(pipeline, status, show_camera_ui)
                if not _read_keys(pipeline):
                    raise KeyboardInterrupt

            if (pipeline.exit_on_source_end and status['source_finished']):
                print("[Main] EEG source finished — stopping.")
                break
            if run_duration > 0 and (time.time() - start_time) >= run_duration:
                print(f"[Main] Reached run_duration ({run_duration:g}s) — stopping.")
                break

            elapsed = time.time() - tick
            sleep_time = loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Main] Stopping system...")
    finally:
        for event in pipeline.close():
            print(f"[Main] {event}")
        if not headless:
            cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CereCe MI + Gaze wheelchair controller (any module subset).")
    parser.add_argument('--config_path', '--config', dest='config_path',
                        default='config.yaml', help='Config file path')
    parser.add_argument('--modules', type=str, default=None,
                        help='Run ONLY these modules, e.g. "eeg,gaze" '
                             '(aliases: eeg, gaze, sonar, control)')
    parser.add_argument('--enable', type=str, default=None,
                        help='Turn additional modules on (comma separated)')
    parser.add_argument('--disable', type=str, default=None,
                        help='Turn modules off (comma separated)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Force the controller into debug mode (no motor commands)')
    parser.add_argument('--headless', action='store_true',
                        help='No window and no keyboard input')
    parser.add_argument('--duration', type=float, default=None,
                        help='Stop after N seconds (default: run until quit)')
    parser.add_argument('--source', choices=['file', 'device'], default=None,
                        help='Override the EEG input source')
    parser.add_argument('--warmup', type=float, default=None,
                        help='EEG warmup seconds before the first health check')
    parser.add_argument('--list-modules', action='store_true',
                        help='Show module aliases and exit')
    args = parser.parse_args(argv)

    if args.list_modules:
        _list_modules(load_config(args.config_path))
        return 0

    try:
        modules = parse_module_list(args.modules) if args.modules else None
        enable = parse_module_list(args.enable) if args.enable else None
        disable = parse_module_list(args.disable) if args.disable else None
    except ValueError as exc:
        parser.error(str(exc))

    start_MI_Tracking(
        config_path=args.config_path,
        modules=modules,
        enable=enable,
        disable=disable,
        dry_run=args.dry_run,
        headless=args.headless,
        duration=args.duration,
        source=args.source,
        warmup=args.warmup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
