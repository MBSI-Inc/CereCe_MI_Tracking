import argparse
from mi_tracker import MI_Tracker
from evidence_accumulator import Evidence_Accumulator
from wheelchair_controller import Wheelchair_Controller

def start_MI_Tracking(config):
    '''
    Thread setup (rates & shared data)
    1) [MI_Tracker] Hardware receive thread @ 250 Hz: read EEG from the device and push into a ring buffer (write EEG_Buffer).
    2) [MI_Tracker] Processing thread ~ 20 Hz (every 50 ms): pull the latest window from EEG_Buffer -> preprocess -> run MI prediction (read EEG_Buffer, write MI_Command).
    3) [start_MI_Tracking] Wheelchair control thread (as needed): continuously read MI_Command and drive the motors in real time (read MI_Command).
    '''
    mi_tracker = MI_Tracker(config)
    evidence_accumulator = Evidence_Accumulator(config)
    wheelchair_controller = Wheelchair_Controller(config)

    while True:
        # get the [left/right] MI signal, which is predicted using EEG signals 
        mi_signal = mi_tracker.get_MI_signal()  # mi_signal = [left/right/stop, probability]

        # accumulate [left or right] evidence over time to make a robust decision
        evidence = evidence_accumulator.update(mi_signal)

        if evidence == 'left':
            print("Move wheelchair left")
            wheelchair_controller.move_left()
        elif evidence == 'right':
            print("Move wheelchair right")
            wheelchair_controller.move_right()
        elif evidence == 'stop':
            print("No MI detected, stop wheelchair")
            wheelchair_controller.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the facemesh gaze tracker.")
    parser.add_argument('--config', type=str, default='config.json', help='Config file')
    args = parser.parse_args()

    # start MI tracking and control the wheelchair via EEG signals
    start_MI_Tracking(args.config)