import subprocess
import time
import shutil

class Wheelchair_Controller:
    """
    Control class for the wheelchair motors using jrk2cmd.
    
    Refactored from legacy 'a.py' to fit the new architecture.
    Provides high-level commands (move_left, move_right, stop) used by main.py.
    """
    DEFAULT_SPEED = 1.0
    
    def __init__(self, config):
        self.config = config
        
        # --- Motor Configuration ---
        # IDs from legacy code or config
        self.LEFT_MOTOR_ID = str(config.get('left_motor_id', '00408146'))
        self.RIGHT_MOTOR_ID = str(config.get('right_motor_id', '00453644'))

        print(self.LEFT_MOTOR_ID)
        print(self.RIGHT_MOTOR_ID)
        
        # Parameters
        self.base_speed = config.get('base_speed', 1.0) # Speed multiplier (0.0 to 1.0)
        self.debug_mode = config.get('debug_mode', True) # Default to True for safety if not specified
        
        # Constants from legacy move_wheel
        self.NEUTRAL_TARGET = config.get('neutral_target', 2048)
        self.SPEED_SCALE = config.get('speed_scale', 200) # Maps -1.0..1.0 speed to roughly +/- 200 units around 2048
        
        # Speed state variables
        self.left_speed = config.get('left_speed', self.DEFAULT_SPEED)
        self.right_speed = config.get('right_speed', self.DEFAULT_SPEED)
        self.jrk2cmd_available = shutil.which('jrk2cmd') is not None

        print(f"[Wheelchair_Controller] Initialized. Debug Mode: {self.debug_mode}")
        if not self.jrk2cmd_available:
            print('[Controller] Warning: jrk2cmd not found in PATH. Hardware commands will not be sent.')
        
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
        self._move_wheel(speed= -1.0 * self.left_speed, is_right_wheel=False) 
        self._move_wheel(speed= 1.0 * self.right_speed, is_right_wheel=True)

    def move_right(self):
        """
        Turn Right:
        Legacy Logic (key 'd'):
            move_wheel(1, False)  -> Left Wheel Forwards
            move_wheel(-1, True)  -> Right Wheel Backwards
        """
        # Left wheel Forward (1.0), Right wheel Backward (-1.0)
        self._move_wheel(speed= 1.0 * self.left_speed, is_right_wheel=False)
        self._move_wheel(speed= -1.0 * self.right_speed, is_right_wheel=True)

    def move_forward(self):
        self._move_wheel(speed= -1.0 * self.left_speed, is_right_wheel=False)
        self._move_wheel(speed= -1.0 * self.right_speed, is_right_wheel=True)

    def move_backward(self):
        self._move_wheel(speed= 1.0 * self.left_speed, is_right_wheel=False)
        self._move_wheel(speed= 1.0 * self.right_speed, is_right_wheel=True)

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
        clamped_speed = max(-1.5, min(1.5, speed))
        
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
        print(f':speed:{speed}, is_right{is_right_wheel}')

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
        if not self.jrk2cmd_available:
            print(f"[Controller] Skipping hardware command because jrk2cmd is unavailable: {' '.join(cmd)}")
            return

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"[Controller] Error sending command to device {device_id}")
        except FileNotFoundError:
            print("[Controller] 'jrk2cmd' not found.")
        except Exception as e:
            print(f"[Controller] Unexpected error: {e}")

    def _verify_connection(self, device_id):
        """Lightweight check if device exists."""
        if not self.jrk2cmd_available:
            print(f"[Controller] Skipping status check for {device_id}: jrk2cmd unavailable.")
            return

        try:
            subprocess.run(['jrk2cmd', '--device', str(device_id), '--status'], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f'{e}')
            print(f"[Controller] Warning: Device {device_id} not responding.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wheelchair_Controller manual check (dry-run unless --live).")
    parser.add_argument('--live', action='store_true',
                        help='actually send jrk2cmd commands (default: debug prints only)')
    parser.add_argument('--speed-step', type=float, default=0.1)
    args = parser.parse_args()

    ctrl = Wheelchair_Controller({'debug_mode': not args.live})
    print('Commands: w=forward s=backward a=left d=right q=stop '
          '1/2=left speed -/+ 9/0=right speed -/+ r=reset e=exit')

    while True:
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == 'w':
            ctrl.move_forward()
        elif choice == 's':
            ctrl.move_backward()
        elif choice == 'a':
            ctrl.move_left()
        elif choice == 'd':
            ctrl.move_right()
        elif choice == 'q':
            ctrl.stop()
        elif choice == '1':
            ctrl.left_speed = max(0.0, ctrl.left_speed - args.speed_step)
            print(f"Left speed = {ctrl.left_speed:.2f}")
        elif choice == '2':
            ctrl.left_speed = min(5.0, ctrl.left_speed + args.speed_step)
            print(f"Left speed = {ctrl.left_speed:.2f}")
        elif choice == '9':
            ctrl.right_speed = max(0.0, ctrl.right_speed - args.speed_step)
            print(f"Right speed = {ctrl.right_speed:.2f}")
        elif choice == '0':
            ctrl.right_speed = min(5.0, ctrl.right_speed + args.speed_step)
            print(f"Right speed = {ctrl.right_speed:.2f}")
        elif choice == 'r':
            ctrl.left_speed = Wheelchair_Controller.DEFAULT_SPEED
            ctrl.right_speed = Wheelchair_Controller.DEFAULT_SPEED
            print('Speeds reset')
        elif choice == 'e':
            break

    ctrl.stop()
    print("Good bye.")
