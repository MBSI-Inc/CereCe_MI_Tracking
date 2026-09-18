"""
Headless smoke runner: boot any subset of the pipeline modules and report
pass/fail without opening a window, and (by default) without touching hardware.

    python -m tools.smoke_run --module eeg --duration 5
    python -m tools.smoke_run --module eeg,gaze --source file --fake
    python -m tools.smoke_run --all --duration 3
    python -m tools.smoke_run --module gaze --real --dry-run

Fake mode (the default) injects tests/fakes.py, so no camera, EEG, BLE sensor
or wheelchair is required. Real mode uses the actual modules; it always runs
dry (debug controller) unless `--live` is passed explicitly.
"""

import argparse
import itertools
import sys
import time
import traceback

from modules.pipeline import (
    MODULE_ALIASES,
    MODULE_KEYS,
    Pipeline,
    parse_module_list,
    resolve_modules,
)

try:
    from tests.fakes import base_config, fake_factories
except ImportError as exc:  # pragma: no cover - only hit in odd deployments
    base_config = fake_factories = None
    _FAKES_IMPORT_ERROR = exc
else:
    _FAKES_IMPORT_ERROR = None


def _load_config(path):
    """Load a YAML config, or fall back to built-in defaults."""
    try:
        from utils.load_config import load_config
        return load_config(path)
    except Exception as exc:
        print(f"[smoke] could not load '{path}' ({exc}) — using built-in defaults")
        return {}


def _run_subset(config, factories, enabled_keys, duration, loop_interval,
                warmup, fake):
    """Boot one module subset and tick it for `duration` seconds."""
    result = {
        'modules': [k for k in MODULE_KEYS if k in enabled_keys],
        'ok': False,
        'ticks': 0,
        'commands': [],
        'degraded': {},
        'detail': '',
    }

    pipeline = None
    try:
        pipeline = Pipeline(
            config,
            modules=resolve_modules(config.get('modules'), selection=enabled_keys),
            dry_run=not fake,
            factories=factories,
            warmup_seconds=warmup,
        )
        pipeline.start()
        result['degraded'] = dict(pipeline.degraded)

        deadline = time.time() + duration
        status = None
        while time.time() < deadline:
            status = pipeline.step()
            result['ticks'] += 1
            if status['command'] is not None and (
                    not result['commands'] or result['commands'][-1] != status['command']):
                result['commands'].append(status['command'])
            if pipeline.exit_on_source_end and status['source_exhausted']:
                break
            time.sleep(loop_interval)

        if result['ticks'] == 0:
            result['detail'] = 'no ticks executed'
            return result

        # A gaze-driven subset with the controller must actually command motion.
        expects_motion = ('gaze_receiver' in result['modules']
                          and 'wheelchair_controller' in result['modules']
                          and 'eeg_receiver' not in result['modules'])
        if expects_motion and not any(c != 'stop' for c in result['commands']):
            result['detail'] = ('expected a movement command from gaze, '
                                'saw only stop')
            return result

        # Motors must be stopped on the way out.
        close_events = pipeline.close()
        pipeline = None
        stopped = any('Motors stopped' in e for e in close_events)
        if result['modules'] and 'wheelchair_controller' in result['modules'] and not stopped:
            result['detail'] = 'motors were not stopped on close'
            return result

        result['ok'] = True
        return result
    except Exception as exc:
        result['detail'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
        return result
    finally:
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass


def _subsets(module_aliases):
    keys = [MODULE_ALIASES[a] for a in module_aliases]
    for size in range(1, len(keys) + 1):
        for combo in itertools.combinations(keys, size):
            yield set(combo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--module', action='append', default=None,
                        help='Module subset, e.g. "eeg" or "eeg,gaze". Repeatable.')
    parser.add_argument('--all', action='store_true',
                        help='Run every non-empty subset of the four modules')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Seconds to tick each subset (default 3)')
    parser.add_argument('--config_path', '--config', dest='config_path',
                        default='config.yaml')
    parser.add_argument('--source', choices=['file', 'device'], default=None,
                        help='Force the EEG input source (file mode works without hardware)')
    parser.add_argument('--real', action='store_true',
                        help='Use real modules instead of tests/fakes.py')
    parser.add_argument('--fake', action='store_true',
                        help='Use fake modules (default)')
    parser.add_argument('--live', action='store_true',
                        help='With --real, allow real motor commands (default is dry)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print tracebacks for failures')
    args = parser.parse_args(argv)

    fake = not args.real
    if args.fake and args.real:
        parser.error('--fake and --real are mutually exclusive')

    if fake and _FAKES_IMPORT_ERROR is not None:
        print(f"[smoke] fake mode unavailable: {_FAKES_IMPORT_ERROR}")
        print("[smoke] rerun with --real, or ensure tests/fakes.py is importable.")
        return 2

    config = _load_config(args.config_path)
    if args.source:
        config.setdefault('receiver_params', {})['input_mode'] = args.source
    if not args.live:
        config.setdefault('control_params', {})['debug_mode'] = True
    config['exit_on_source_end'] = False
    warmup = 0.0 if fake else float(config.get('eeg_warmup_seconds', 5.0))
    loop_interval = float(config.get('loop_interval', 0.05))

    if args.all:
        subsets = list(_subsets(list(MODULE_ALIASES)))
    elif args.module:
        subsets = []
        for spec in args.module:
            try:
                keys = set(parse_module_list(spec))
            except ValueError as exc:
                parser.error(str(exc))
            if keys not in subsets:
                subsets.append(keys)
    else:
        subsets = [set(MODULE_ALIASES.values())]

    factories = fake_factories() if fake else None
    if fake:
        params = {k: v for k, v in config.items()
                  if k in ('receiver_params', 'predictor_params',
                           'accumulator_params', 'control_params',
                           'gaze_params', 'ultrasonic_params')}
        # Script the fakes so the smoke run exercises real state transitions
        # rather than sitting in the idle default.
        params['receiver_params'] = {**params.get('receiver_params', {}),
                                     'healthy': True, 'input_mode': 'file'}
        params['accumulator_params'] = {**params.get('accumulator_params', {}),
                                        'activate_at': 1}
        params['gaze_params'] = {**params.get('gaze_params', {}),
                                 'directions': ['forward']}
        params['ultrasonic_params'] = {**params.get('ultrasonic_params', {}),
                                       'available': True, 'obstacle': False}
        config = base_config(**params)

    mode = 'FAKE' if fake else ('REAL+dry' if not args.live else 'REAL+LIVE')
    print(f"[smoke] mode={mode} config={args.config_path} "
          f"duration={args.duration:g}s subset(s)={len(subsets)}")
    print()

    results = []
    for keys in subsets:
        label = ','.join(sorted(keys))
        print(f"[smoke] --- {label} ---")
        outcome = _run_subset(config, factories, keys, args.duration,
                              loop_interval, warmup, fake)
        results.append(outcome)
        status = 'PASS' if outcome['ok'] else 'FAIL'
        print(f"[smoke] {status} ticks={outcome['ticks']} "
              f"commands={outcome['commands']}"
              + (f" degraded={list(outcome['degraded'])}" if outcome['degraded'] else ''))
        if outcome['detail']:
            print(f"[smoke]      {outcome['detail']}")
            if args.verbose and outcome.get('traceback'):
                print(outcome['traceback'])
        print()

    passed = sum(1 for r in results if r['ok'])
    print(f"[smoke] {passed}/{len(results)} subsets passed")
    for r in results:
        if not r['ok']:
            print(f"[smoke]   FAILED: {','.join(sorted(r['modules']))} — {r['detail']}")
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
