import serial
import time

class ultrasound:
    def __init__(self, port: str, baud_rate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            time.sleep(2)  # Allow Arduino to reset
            self.ser.reset_input_buffer()
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to open serial port {self.port}: {e}")

    def get_distance(self) -> float:
        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            try:
                return float(line)
            except ValueError:
                pass  # Skip malformed lines

        return None  # No valid data available

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


