### Real-time Control Challenges & Solutions

![architecture diagram](./figs/architecture_v3.png)



**1. Sampling Rate Mismatch**
- **Problem:** The EEG hardware streams data (250Hz) faster than the inference or motor control loops can process, leading to potential lag or backlog.
- **Solution:** **Rolling FIFO Buffer**. We utilize a rolling FIFO buffer to maintain the most current EEG signals. The predictor asynchronously retrieves the latest data chunk from this buffer to forecast MI signals and control the wheelchair. The operating frequency depends on the pipeline's processing speed, ensuring minimal latency.

**2. Signal Instability**
- **Problem:** Raw EEG predictions fluctuate due to noise, causing erratic wheelchair movements (jitter) or false positives.
- **Solution:** **Evidence Accumulation**. We implement a continuous integrator (Leaky Integrate-and-Fire). Movement commands are only triggered when the accumulated confidence score exceeds a robust threshold, effectively smoothing out transient noise.
- **Notes**: Turning requires strong and continuous mental concentration

