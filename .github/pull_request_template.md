## What changed

<!-- Summary. Link the issue this closes: Closes #N -->

## Why

## Platform verification

We develop this on two platforms and neither side can fully test the other's
code paths (the Linux Bluetooth shim, the Windows ExplorePy/dashboard paths).
Fill this in honestly — an untested box is useful information, not a failure.

- [ ] Linux — tested
- [ ] Windows — tested
- [ ] Needs verification on the other platform → issue: #
- [ ] Not platform-specific

## Test evidence

<!--
Paste the commands you ran and their real output. For reference, the checks we
expect to pass:

  pytest
  python -m tools.smoke_run --all --duration 3
  python -m modules.eeg_receiver --mode file --data <csv>
  MI_TEST_FILE=<short csv> python main_test.py

-->

```
```

## Hardware used

<!-- e.g. Explore_842F headset, UltrasonicBLE Arduino, Jrk controllers, camera. Say "none" if dry-run. -->

## Checklist

- [ ] `pytest` passes locally
- [ ] No new direct pushes to a shared branch (this PR is the review point)
- [ ] Config changes are reflected in `config.yaml` and the mode configs
- [ ] `README.md` updated if behaviour or flags changed
- [ ] Motors/actuators were not driven without an explicit `--live` / `debug_mode: false`
