import os
import sys
from typing import Optional

# Add Cerebruh_Gaze_Tracking/src to path so GazeTracker and its util can be imported
_gaze_src = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Cerebruh_Gaze_Tracking', 'src')
)
if _gaze_src not in sys.path:
    sys.path.insert(0, _gaze_src)

from facemesh_gazetracker import GazeTracker


class Gaze_Receiver:
    """
    Wraps GazeTracker (head-pose mode) and exposes a clean interface
    consistent with the rest of the CereCe_MI_Tracking module architecture.

    Expected config keys:
        config_path (str):              Path to the GazeTracker JSON config file.
        horizontal_threshold (int):     |horizontal| > threshold → left/right turn.  Default 50.
        vertical_forward_threshold (int): vertical > threshold → forward.            Default 30.
        vertical_backward_threshold (int): vertical < threshold → backward.          Default 0.
    """

    def __init__(self, config: dict):
        config_path = config.get('config_path', 'Cerebruh_Gaze_Tracking/config.json')
        self.h_threshold  = config.get('horizontal_threshold', 50)
        self.v_forward    = config.get('vertical_forward_threshold', 30)
        self.v_backward   = config.get('vertical_backward_threshold', 0)

        print(f"[Gaze_Receiver] Loading GazeTracker from: {config_path}")
        self.tracker = GazeTracker(config_path)
        print("[Gaze_Receiver] Initialized.")

    def start(self):
        """Start the background tracking thread."""
        self.tracker.start()
        print("[Gaze_Receiver] Tracking thread started.")

    def stop(self):
        """Stop the background tracking thread."""
        self.tracker.stop()
        print("[Gaze_Receiver] Tracking thread stopped.")

    def get_raw(self) -> Optional[dict]:
        """Return the latest raw gaze output dict, or None if not yet available."""
        return self.tracker.get_gaze_async()

    def get_direction(self) -> Optional[str]:
        """
        Translate the latest gaze output into a direction command.

        Returns:
            'forward' | 'backward' | 'left' | 'right' | 'stop'
            None  — if no data has been produced yet (tracker still warming up).

        Priority: horizontal gaze takes precedence over vertical so that diagonal
        head poses are resolved as a turn rather than a forward/backward move.
        """
        data = self.get_raw()
        if data is None:
            return None

        if not data.get('face_detected', False):
            return 'stop'

        h = data['combined_gaze_horizontal']
        v = data['combined_gaze_vertical']

        if h < -self.h_threshold:
            return 'left'
        elif h > self.h_threshold:
            return 'right'
        elif v > self.v_forward:
            return 'forward'
        elif v < self.v_backward:
            return 'backward'
        else:
            return 'stop'


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test  —  run with:  python modules/gaze_receiver.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    _HERE         = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, '..'))

    # ── Configurable test parameters ─────────────────────────────────────────
    TEST_CONFIG = {
        'config_path':                os.path.join(_PROJECT_ROOT, 'config_gaze.json'),
        'horizontal_threshold':       50,
        'vertical_forward_threshold': 30,
        'vertical_backward_threshold': 0,
    }
    WARMUP_SECONDS  = 1.5   # wait before first validity check
    SAMPLE_SECONDS  = 8.0   # total live-sampling window
    PRINT_INTERVAL  = 0.5   # how often to print a live line

    VALID_DIRECTIONS = {'forward', 'backward', 'left', 'right', 'stop', None}
    REQUIRED_KEYS    = {
        'combined_gaze_horizontal', 'combined_gaze_vertical',
        'face_detected', 'move_enabled', 'blinked',
    }

    # ── Helpers ───────────────────────────────────────────────────────────────
    GREEN = '\033[32m'
    RED   = '\033[31m'
    RESET = '\033[0m'

    results = []   # list of bool

    def check(name, ok, detail=''):
        tag = f'{GREEN}PASS{RESET}' if ok else f'{RED}FAIL{RESET}'
        print(f'  [{tag}] {name}' + (f'  —  {detail}' if detail else ''))
        results.append(ok)

    # ── Test body ─────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print('  Gaze_Receiver — standalone integration test')
    print(f'  config : {TEST_CONFIG["config_path"]}')
    print('=' * 60)

    # Step 1 ── init
    print('\n[1/5] Initialize')
    gaze = None
    try:
        gaze = Gaze_Receiver(TEST_CONFIG)
        check('Gaze_Receiver.__init__()', True)
    except Exception as exc:
        check('Gaze_Receiver.__init__()', False, str(exc))
        print('\n  Cannot continue — aborting.')
        sys.exit(1)

    # Step 2 ── start thread
    print('\n[2/5] Start background tracking thread')
    try:
        gaze.start()
        check('gaze.start()', True)
    except Exception as exc:
        check('gaze.start()', False, str(exc))
        sys.exit(1)

    # Step 3 ── warmup
    print(f'\n[3/5] Warmup ({WARMUP_SECONDS}s) — waiting for first frame...')
    time.sleep(WARMUP_SECONDS)
    first = gaze.get_raw()
    check(
        'First frame received after warmup',
        first is not None,
        'get_raw() still None — check camera / MediaPipe' if first is None else '',
    )

    # Step 4 ── live sampling
    print(f'\n[4/5] Live output ({SAMPLE_SECONDS}s) — move your head to verify mapping')
    print(f'  {"t(s)":>5}  {"H-Gaze":>7}  {"V-Gaze":>7}  {"Face":>5}  {"Yaw°":>6}  {"Pitch°":>7}  Direction')
    print('  ' + '─' * 58)

    samples     = []   # list of (raw_dict, direction_str)
    t0          = time.time()
    t_last_row  = -PRINT_INTERVAL   # force immediate first row

    while time.time() - t0 < SAMPLE_SECONDS:
        raw  = gaze.get_raw()
        dirn = gaze.get_direction()
        now  = time.time() - t0

        if raw is not None:
            samples.append((raw, dirn))

            if now - t_last_row >= PRINT_INTERVAL:
                h     = raw.get('combined_gaze_horizontal', '?')
                v     = raw.get('combined_gaze_vertical',   '?')
                face  = raw.get('face_detected', '?')
                yaw   = raw.get('_yaw',   '?')
                pitch = raw.get('_pitch', '?')
                yaw_s   = f'{yaw:+.1f}'   if isinstance(yaw,   float) else str(yaw)
                pitch_s = f'{pitch:+.1f}' if isinstance(pitch, float) else str(pitch)
                print(f'  {now:>5.1f}s  {str(h):>7}  {str(v):>7}  {str(face):>5}  '
                      f'{yaw_s:>6}  {pitch_s:>7}  {dirn}')
                t_last_row = now

        time.sleep(0.05)

    # Step 5 ── validate collected samples
    n = len(samples)
    print(f'\n[5/5] Validate ({n} frames collected)')

    if n == 0:
        check('Frames received', False, 'No frames — is a camera connected?')
    else:
        check('Frames received', True, f'{n} frames')

        # Required keys
        missing_ever = set()
        for raw, _ in samples:
            missing_ever |= REQUIRED_KEYS - set(raw.keys())
        check('All required keys present in every frame', not missing_ever,
              f'Missing: {missing_ever}' if missing_ever else '')

        # Numeric ranges
        h_vals = [r['combined_gaze_horizontal'] for r, _ in samples
                  if isinstance(r.get('combined_gaze_horizontal'), (int, float))]
        v_vals = [r['combined_gaze_vertical']   for r, _ in samples
                  if isinstance(r.get('combined_gaze_vertical'),   (int, float))]

        h_ok = bool(h_vals) and all(-100 <= x <= 100 for x in h_vals)
        v_ok = bool(v_vals) and all(-100 <= x <= 100 for x in v_vals)
        check('combined_gaze_horizontal in [-100, 100]', h_ok,
              f'range seen [{min(h_vals)}, {max(h_vals)}]' if h_vals else 'no data')
        check('combined_gaze_vertical   in [-100, 100]', v_ok,
              f'range seen [{min(v_vals)}, {max(v_vals)}]' if v_vals else 'no data')

        # Direction validity
        bad = [(r, d) for r, d in samples if d not in VALID_DIRECTIONS]
        check('get_direction() returns only valid values', not bad,
              f'Invalid: {set(d for _, d in bad)}' if bad else '')

        # face_detected type
        type_ok = all(isinstance(r.get('face_detected'), bool) for r, _ in samples)
        check('face_detected is bool', type_ok)

        # Direction distribution summary
        dist = {}
        for _, d in samples:
            dist[str(d)] = dist.get(str(d), 0) + 1
        print(f'\n  Direction distribution across {n} frames:')
        for dirn, cnt in sorted(dist.items(), key=lambda x: -x[1]):
            bar = '█' * int(cnt / n * 30)
            print(f'    {dirn:>10}  {cnt:>4}  {bar}')

    # Stop tracker
    gaze.stop()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    failed = len(results) - passed
    print()
    print('=' * 60)
    if failed == 0:
        print(f'  {GREEN}ALL {passed} CHECKS PASSED{RESET}')
    else:
        print(f'  {RED}{failed} / {len(results)} CHECKS FAILED{RESET}')
    print('=' * 60)
    print()

    sys.exit(0 if failed == 0 else 1)


