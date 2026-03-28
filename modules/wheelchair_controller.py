import subprocess
import time

class Wheelchair_Controller:
    """
    Control class for the wheelchair motors using jrk2cmd.
    
    Refactored from legacy 'a.py' to fit the new architecture.
    Provides high-level commands (move_left, move_right, stop) used by main.py.
    """
    
    def __init__(self, config):
        self.config = config
        
        # --- Motor Configuration ---
        # IDs from legacy code or config
        self.LEFT_MOTOR_ID = str(config.get('left_motor_id', '00408146'))
        self.RIGHT_MOTOR_ID = str(config.get('right_motor_id', '00453644'))
        
        # Parameters
        self.base_speed = config.get('base_speed', 1.0) # Speed multiplier (0.0 to 1.0)
        self.debug_mode = config.get('debug_mode', True) # Default to True for safety if not specified
        
        # Constants from legacy move_wheel
        self.NEUTRAL_TARGET = config.get('neutral_target', 2048)
        self.SPEED_SCALE = config.get('speed_scale', 200) # Maps -1.0..1.0 speed to roughly +/- 200 units around 2048
        
        print(f"[Wheelchair_Controller] Initialized. Debug Mode: {self.debug_mode}")
        
        # Ensure devices are reachable (Optional check on startup)
        if not self.debug_mode:
            self._verify_connection(self.LEFT_MOTOR_ID)
            self._verify_connection(self.RIGHT_MOTOR_ID)

    def move_left(self):
        """
        Turn Left:
        Legacy Logic (key 'a'):
            move_wheel(-1, False) -> Left Wheel Backwards
            move_wheel(1, True)   -> Right Wheel Forwards
        """
        # Left wheel Backward (-1.0), Right wheel Forward (1.0)
        self._move_wheel(speed=-1.0, is_right_wheel=False) 
        self._move_wheel(speed=1.0, is_right_wheel=True)

    def move_right(self):
        """
        Turn Right:
        Legacy Logic (key 'd'):
            move_wheel(1, False)  -> Left Wheel Forwards
            move_wheel(-1, True)  -> Right Wheel Backwards
        """
        # Left wheel Forward (1.0), Right wheel Backward (-1.0)
        self._move_wheel(speed=1.0, is_right_wheel=False)
        self._move_wheel(speed=-1.0, is_right_wheel=True)

    def stop(self):
        """
        Stop motors.
        Sets target to NEUTRAL (2048) to actively hold/stop position.
        This provides smoother control than disabling power.
        """
        if self.debug_mode:
            print("[DEBUG] Stopping Motors (Target -> Neutral)")
        else:
            self._send_jrk_cmd(self.LEFT_MOTOR_ID, '--target', str(self.NEUTRAL_TARGET))
            self._send_jrk_cmd(self.RIGHT_MOTOR_ID, '--target', str(self.NEUTRAL_TARGET))

    def _move_wheel(self, speed: float, is_right_wheel: bool):
        """
        Internal helper to execute movement command for a single wheel.
        """
        # Clamp input speed
        clamped_speed = max(-1.0, min(1.0, speed))
        
        # Apply base speed multiplier
        final_speed = clamped_speed * self.base_speed
        
        # Calculate target value
        # Base: 2048
        # Range: +/- 200 * speed
        # Note: Legacy code had `if not is_right: speed = -speed`. This suggests the left motor
        # is physically mounted inversely or wired inversely relative to "forward" command.
        # However, `move_left` logic above explicitly passes -1.0 to left wheel.
        
        # Let's verify legacy logic 'a.py':
        # def move_wheel(speed, is_right):
        #    ...
        #    if not is_right: speed = -speed
        #    calculated_speed = int(2048 + (speed * 200))
        
        # If I call move_left -> _move_wheel(-1.0, False)
        # Inside legacy: is_right=False -> speed becomes -(-1.0) = 1.0
        # Target = 2048 + 200 = 2248.
        # Wait, if Left wheel target 2248 means Forward, then `move_left` (Left Back, Right Fwd) logic in legacy was:
        # key 'a': move_wheel(-1, False); move_wheel(1, True)
        # Left: -1 input -> becomes 1 (2248) -> Forward?
        # Right: 1 input -> becomes 1 (2248) -> Forward?
        # That would mean key 'a' moves both Forward? No, that's wrong for a turn.
        
        # Let's re-read legacy 'a.py' carefully.
        # key 'w' (Forward): move_wheel(1, False), move_wheel(1, True)
        #   Left (1, False): speed -> -1 -> 1848 (Backward?)
        #   Right (1, True): speed -> 1  -> 2248 (Forward?)
        #   Result: Left Back, Right Fwd?? This spins the chair!
        
        # Maybe the physical mounting is:
        # Left Motor 1848 = Forward, 2248 = Backward?
        # Right Motor 2248 = Forward, 1848 = Backward?
        
        # IF key 'w' works as forward in legacy, then:
        # Left(1.0) -> Target 1848.
        # Right(1.0) -> Target 2248.
        
        # Now key 'a' (Left Turn):
        # move_wheel(-1, False) -> speed(1) -> Target 2248 (Backward relative to fw motion?)
        # move_wheel(1, True)   -> speed(1) -> Target 2248 (Forward)
        # Result: Left 2248 (Back), Right 2248 (Fwd). -> Spins Left. Correct.
        
        # So my `_move_wheel` implementation must replicate the target calculation exactly.
        
        # Replicating logic:
        effective_speed = final_speed
        if not is_right_wheel:
            effective_speed = -effective_speed
            
        calculated_target = int(self.NEUTRAL_TARGET + (effective_speed * self.SPEED_SCALE))
        
        if self.debug_mode:
            side = "Right" if is_right_wheel else "Left"
            print(f"[DEBUG] {side} Wheel | Input: {speed} | Target: {calculated_target}")
        else:
            device_id = self.RIGHT_MOTOR_ID if is_right_wheel else self.LEFT_MOTOR_ID
            self._send_jrk_cmd(device_id, '--target', str(calculated_target))


    def _send_jrk_cmd(self, device_id, *args):
        """Run jrk2cmd via subprocess."""
        cmd = ['jrk2cmd', '--device', str(device_id)] + list(args)
        try:
            # Using check_call or run to avoid blocking too long/capturing huge output
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"[Controller] Error sending command to device {device_id}")
        except FileNotFoundError:
            print("[Controller] 'jrk2cmd' not found.")
        except Exception as e:
            print(f"[Controller] Unexpected error: {e}")

    def _verify_connection(self, device_id):
        """Lightweight check if device exists."""
        try:
            subprocess.run(['jrk2cmd', '--device', str(device_id), '--status'], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            print(f"[Controller] Warning: Device {device_id} not responding.")

if __name__ == "__main__":
    # Unit Test intended for reconstruction folder execution
    print("--- Testing Wheelchair Controller (Debug Mode) ---")
    
    config = {
        'debug_mode': False, 
        'left_motor_id': 'TEST_L', 
        'right_motor_id': 'TEST_R'
    }

    ctrl = Wheelchair_Controller(config)

    print('Commands: 1=Left Turn, 2=Right Turn, 3=Stop, 4=Exit')

    while True:
        choice = input("> ")
        choice = choice.lower() #Convert input to "lowercase"

        if choice == 'exit':
            print("Good bye.")
            break

        if choice == '1':
            print("\n1. Testing Left Turn")
            ctrl.move_left()

        if choice == '2':
            print("\n2. Testing Right Turn")
            ctrl.move_right()

        if choice == '3':
            print("\n3. Testing Stop")
            ctrl.stop()

        if choice == '4':
            print("Good bye.")
            break
        else:
            continue
    
    
    
   
    
   
    
    
