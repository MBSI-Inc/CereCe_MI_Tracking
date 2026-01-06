### Real-time Control Challenges & Solutions

**1. Sampling Rate Mismatch**
* **Problem:** The EEG hardware streams data (250Hz) faster than the inference or motor control loops can process, leading to potential lag or backlog.
* **Solution:** **Last Value Caching (LVC)**. We use a shared state variable that always holds the *single most recent* command. The controller polls the latest data and ignores obsolete intermediate states, ensuring zero latency.

**2. Signal Instability**
* **Problem:** Raw EEG predictions fluctuate due to noise, causing erratic wheelchair movements (jitter) or false positives.
* **Solution:** **Evidence Accumulation**. We implement a continuous integrator (Leaky Integrate-and-Fire). Movement commands are only triggered when the accumulated confidence score exceeds a robust threshold, effectively smoothing out transient noise.


### main.py
- Start mi_tracker

- Continuously control the wheelchair using MI signal from mi_tracker until the program ends

### mi_tracker pipeline:
- connect to an EEG device (e.g., OpenBCI, Muse, etc.)

- preprocess EEG data (e.g., bandpass filtering, artifact removal, etc.)

- compute power spectral density (PSD) features from EEG data

- use the pre-trained MI model to predict MI signals (left/right/no MI)

- return the predicted MI signal for wheelchair control


### threads
-  [MI_Tracker.py] Hardware receive thread @ 250 Hz: read EEG from the device and push into a ring buffer (write EEG_Buffer).

-  [MI_Tracker.py] Processing thread ~ 20 Hz (every 50 ms): pull the latest window from EEG_Buffer -> preprocess -> run MI prediction (read EEG_Buffer, write MI_Command).

-  [main.py] Wheelchair control thread (as needed): continuously read MI_Command and drive the motors in real time (read MI_Command).