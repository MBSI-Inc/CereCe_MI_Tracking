
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