# Working across Windows and Linux

This project is developed from two platforms, often by two agents working in
parallel. Neither platform can exercise all of the other's code paths — the
Linux Bluetooth socket shim only runs on Linux, the ExplorePy dashboard and
camera paths were first brought up on Windows — so a change that is "done" on
one side can still be broken on the other.

## The rules

1. **Do not push directly to a shared branch** (`feat/unified`, `main`).
   Work on a short topic branch (`feat/<topic>`, `fix/<topic>`) and open a PR.
   Two agents pushing to one branch is what produced a duplicated
   end-of-replay fix in `main_test.py` and `EEG_Receiver`, each solving the
   same problem a different way.
2. **Every PR states which platform it was verified on** using the PR template.
   An untested box is useful information; a silent one is not.
3. **Anything that cannot be verified locally gets an issue**, using the
   *Cross-platform verification request* template, labelled
   `needs-verification` and `platform:linux` or `platform:windows`. The other
   side runs it and reports the real output before the issue closes.
4. **Check the issue list before starting work** so the same fix is not written
   twice from two directions.

## Labels

| Label | Meaning |
|---|---|
| `platform:linux` | Only reproducible or only testable on Linux |
| `platform:windows` | Only reproducible or only testable on Windows |
| `needs-verification` | Implemented, but not yet exercised on the other platform |
| `handoff` | Work being handed to the other platform or tracked across both |

## What to put in a handoff

Be specific enough that the other side does not have to guess:

- the branch and commit,
- the exact commands, including environment quirks (on this machine the project
  env is the `mi-tracking` conda env, not `.venv`),
- the exact expected output — exit code, line counts, timings,
- which hardware was attached, or "none".

The commit messages in this repo are already good at this; the issue is that
nobody can find them later. That is the gap issues fill.

## Running the checks

```bash
pytest                                              # no hardware needed
python -m tools.smoke_run --all --duration 3        # every module subset, fakes
MI_TEST_FILE=data/MItest_24-01-27_ExG_short.csv python main_test.py   # ~31 s replay
python main.py --modules eeg,gaze --dry-run         # real modules, no motors
```

Hardware tests are marked and deselected by default; opt in with
`pytest -m hardware`.

## Safety

Never drive the motors by accident. `--dry-run` and `control_params.debug_mode`
keep the controller in print-only mode, and start-up announces
`*** LIVE MOTOR CONTROL ENABLED ***` when commands will really be sent.
