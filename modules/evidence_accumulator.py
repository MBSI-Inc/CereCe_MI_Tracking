class Evidence_Accumulator:
    """
    Implements a temporal smoothing algorithm to stabilize noisy predictions 
    from the MI Predictor. Accumulates evidence over consecutive frames before 
    triggering a command.
    
    The logic:
    1. Maintain 'evidence' scores for 'left' and 'right'.
    2. At each update:
       - Decay ALL current evidence by a fixed `decay_rate` (e.g. subtract 0.1).
       - If the incoming prediction is valid ('left' or 'right'), INCREASE its evidence by `evidence_build_rate` (e.g. add 1.0).
       - Clamp values between 0 and `max_evidence`.
    3. Determine the command:
       - If the strongest evidence > `threshold`, output that command.
       - Use hysteresis: Once triggered, the command stays active until evidence drops below `hysteresis_threshold` (e.g., 50% of threshold).
       - Otherwise, output 'stop'.
    """

    def __init__(self, config):
        self.config = config
        
        # Accumulation Parameters
        self.threshold = config.get('threshold', 3.0) 
        self.decay = config.get('decay', 0.2)             # Amount to decrease per step
        self.build_rate = config.get('build_rate', 1.0)   # Amount to increase for matching prediction
        self.max_evidence = config.get('max_evidence', 5.0)
        self.hysteresis_factor = config.get('hysteresis_factor', 0.6) # Multiplier for drop-off threshold

        # Internal state
        self.evidence = {
            'left': 0.0,
            'right': 0.0
        }
        self.current_command = 'stop'


    def update(self, raw_prediction):
        """
        Updates the evidence accumulator with a new raw prediction and determines the stable command.

        Args:
            raw_prediction (str): The latest prediction from MI_Predictor ('left', 'right', 'none').

        Returns:
            str: The stabilized command ('left', 'right', 'stop').
        """
        
        # 1. Apply Decay to ALL evidence
        for key in self.evidence:
            self.evidence[key] = max(0.0, self.evidence[key] - self.decay)

        # 2. Add evidence for the current raw prediction
        if raw_prediction in self.evidence:
            # Boost the evidence for the matching class
            self.evidence[raw_prediction] += self.build_rate
            # Clamp to max_evidence
            self.evidence[raw_prediction] = min(self.evidence[raw_prediction], self.max_evidence)

        # 3. Determine command based on thresholds and hysteresis
        
        # Find the class with the strongest evidence
        strongest_cmd = max(self.evidence, key=self.evidence.get)
        strongest_val = self.evidence[strongest_cmd]

        # Calculate dynamic threshold based on current state (Hysteresis)
        # If we are already in a command state, it's harder to leave (lower threshold).
        # If we are in 'stop', we need full threshold to start.
        
        active_threshold = self.threshold * self.hysteresis_factor if self.current_command == strongest_cmd else self.threshold

        if strongest_val >= active_threshold:
            self.current_command = strongest_cmd
        else:
            # If even the strongest evidence is below its active threshold, we stop.
            self.current_command = 'stop'

        return self.current_command


if __name__ == "__main__":
    # --- Integration Test ---
    print("--- Testing Evidence Accumulator ---")
    
    test_config = {
        'threshold': 3.0,
        'decay': 0.2,             # lose 0.2 evidence per frame if not reinforced (if main loop runs at 20hz, this means ~20x0.2=4 evidence lost per second)
        'build_rate': 1.0,        # gain 1.0 evidence per matching frame
        'max_evidence': 5.0,
        'hysteresis_factor': 0.5  # drop below 1.5 to stop
    }
    acc = Evidence_Accumulator(test_config)

    # Sequence simulation:
    # 1. Noise/Stop (should stay stopped)
    # 2. Strong Left input (should trigger 'left' after ~3-4 frames)
    # 3. Intermittent drop (should Stay 'left' due to hysteresis)
    # 4. Long stop (should revert to 'stop')
    
    test_sequence = [
        'none', 'none', 'right', 'none',  # Noise
        'left', 'left', 'left', 'left',   # Strong Left signal (Accumulating...)
        'left', 'none', 'left',           # Signal flicker (Should hold 'left')
        'none', 'none', 'none', 'none',   # Signal loss (Decaying...)
        'none', 'none'
    ]
    
    print(f"{'Input':<10} | {'L Ev':<5} | {'R Ev':<5} | {'Output':<10}")
    print("-" * 40)

    for pred in test_sequence:
        output_cmd = acc.update(pred)
        print(f"{pred:<10} | {acc.evidence['left']:.2f}  | {acc.evidence['right']:.2f}  | {output_cmd:<10}")

    print("--- Test Complete ---")

    print("\n--- Test Case 2: Conflict Switching (Left -> Right) ---")
    # Reset accumulator for clean test
    acc = Evidence_Accumulator(test_config)
    
    conflict_sequence = [
        'left', 'left', 'left', 'left', 'left', # Trigger Left (Ev: ~5.0)
        'right', 'right', 'right',              # Right starts building, Left decays
        'right', 'right', 'right',              # Right overtakes Left?
        'right', 'right', 'right'
    ]

    print(f"{'Input':<10} | {'L Ev':<5} | {'R Ev':<5} | {'Output':<10}")
    print("-" * 40)
    
    for pred in conflict_sequence:
        output_cmd = acc.update(pred)
        print(f"{pred:<10} | {acc.evidence['left']:.2f}  | {acc.evidence['right']:.2f}  | {output_cmd:<10}")

    print("\n--- All Tests Complete ---")
