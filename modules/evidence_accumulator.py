class Evidence_Accumulator:
    """
    Implements a temporal smoothing algorithm to stabilize noisy MI predictions
    before they reach the wheelchair controller.

    The logic:
    1. Maintain 'evidence' scores for 'active' (grip) and 'inactive' (rest).
    2. At each update:
       - Decay ALL current evidence by a fixed `decay_rate` (e.g. subtract 0.2).
       - If the incoming prediction is 'active' or 'inactive', INCREASE its evidence by `build_rate`.
       - Clamp values between 0 and `max_evidence`.
    3. Determine the command:
       - If the strongest evidence > `threshold`, output that command.
       - Use hysteresis: once triggered, the command stays until evidence drops below
         `threshold * hysteresis_factor` — prevents rapid toggling.
       - Otherwise, output 'inactive' (safe default).
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
            'active': 0.0,
            'inactive': 0.0,
        }
        self.current_command = 'inactive'


    def update(self, raw_prediction):
        """
        Updates the evidence accumulator with a new raw prediction and determines the stable command.

        Args:
            raw_prediction (str): The latest prediction from MI_Predictor ('active', 'inactive').

        Returns:
            str: The stabilized command ('active' or 'inactive').
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
        # If already in a state, it's easier to stay (lower threshold).
        # If in 'inactive', full threshold is needed to switch to 'active'.
        
        active_threshold = self.threshold * self.hysteresis_factor if self.current_command == strongest_cmd else self.threshold

        if strongest_val >= active_threshold:
            self.current_command = strongest_cmd
        else:
            # Below threshold — default to safe inactive state.
            self.current_command = 'inactive'

        return self.current_command


if __name__ == "__main__":
    # --- Integration Test ---
    print("--- Testing Evidence Accumulator (active/inactive) ---")

    test_config = {
        'threshold': 3.0,
        'decay': 0.2,
        'build_rate': 1.0,
        'max_evidence': 5.0,
        'hysteresis_factor': 0.5,
    }
    acc = Evidence_Accumulator(test_config)

    # Test Case 1: inactive noise → active signal → flicker → decay back
    test_sequence = [
        'inactive', 'inactive', 'inactive', 'inactive',   # Baseline: should stay inactive
        'active', 'active', 'active', 'active',           # Grip held: should trigger 'active'
        'active', 'inactive', 'active',                   # Signal flicker: should hold 'active' (hysteresis)
        'inactive', 'inactive', 'inactive', 'inactive',   # Grip released: should decay to 'inactive'
        'inactive', 'inactive',
    ]

    print(f"{'Input':<10} | {'Act Ev':<7} | {'Inact Ev':<9} | {'Output':<10}")
    print("-" * 45)

    for pred in test_sequence:
        output_cmd = acc.update(pred)
        print(f"{pred:<10} | {acc.evidence['active']:.2f}   | {acc.evidence['inactive']:.2f}      | {output_cmd:<10}")

    print("--- Test Complete ---")

    print("\n--- Test Case 2: active → inactive switch ---")
    acc = Evidence_Accumulator(test_config)

    switch_sequence = [
        'active', 'active', 'active', 'active', 'active',       # Fully active
        'inactive', 'inactive', 'inactive', 'inactive',          # Release grip
        'inactive', 'inactive', 'inactive', 'inactive',          # Should switch to inactive
    ]

    print(f"{'Input':<10} | {'Act Ev':<7} | {'Inact Ev':<9} | {'Output':<10}")
    print("-" * 45)

    for pred in switch_sequence:
        output_cmd = acc.update(pred)
        print(f"{pred:<10} | {acc.evidence['active']:.2f}   | {acc.evidence['inactive']:.2f}      | {output_cmd:<10}")

    print("\n--- All Tests Complete ---")
