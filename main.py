import argparse
from mi_tracker import MI_Tracker

def start_MI_Tracking(config):
    mi_tracker = MI_Tracker(config)
    while True:
        # get the [left or right] MI signal, which is predicted using EEG signals 
        output = mi_tracker.get_MI_output() 
        if output == 0:
            print("Move wheelchair left")
            # code to move wheelchair left
        elif output == 1:
            print("Move wheelchair right")
            # code to move wheelchair right
        else:
            print("No MI detected, keeping wheelchair still")
            # code to keep wheelchair still


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the facemesh gaze tracker.")
    parser.add_argument('--config', type=str, default='config.json', help='Config file')
    args = parser.parse_args()

    # connect  to the EEG device
    # todo: implement EEG device connection here

    # start MI tracking and control the wheelchair via EEG signals
    start_MI_Tracking(args.config)