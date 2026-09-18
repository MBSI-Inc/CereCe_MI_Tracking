import asyncio
import threading
import time
from typing import Optional, Tuple


class Ultrasonic_Receiver:
    """
    Reads left/right obstacle distances from the UltrasonicBLE Arduino
    (see Ultrasonic_Sensor_Data_Extraction.ino) over BLE in a background
    thread, and exposes them with the same start/stop/get_* shape as the
    other receivers.

    The Arduino only notifies when a distance changes by >0.5 cm, so health
    is tied to the BLE connection rather than to packet recency.

    Expected config keys:
        device_name (str):          BLE advertised name.            Default 'UltrasonicBLE'.
        char_uuid (str):            Notify characteristic UUID.
        obstacle_distance (float):  cm; a reading below this is an obstacle. Default 50.
        scan_timeout (float):       seconds per BLE scan attempt.   Default 5.0.
        reconnect_interval (float): seconds between reconnect tries. Default 3.0.
        block_directions (list):    directions suppressed while an obstacle is
                                    present. Default ['forward'].
    """

    DEFAULT_CHAR_UUID = "a4b5735b-6dd9-4680-ac4b-06b4a209bffa"

    def __init__(self, config: dict):
        self.device_name        = config.get('device_name', 'UltrasonicBLE')
        self.char_uuid          = config.get('char_uuid', self.DEFAULT_CHAR_UUID)
        self.obstacle_distance  = float(config.get('obstacle_distance', 50.0))
        self.scan_timeout       = float(config.get('scan_timeout', 5.0))
        self.reconnect_interval = float(config.get('reconnect_interval', 3.0))
        self.block_directions   = set(config.get('block_directions', ['forward']))

        self._lock = threading.Lock()
        self._lhs: Optional[float] = None
        self._rhs: Optional[float] = None
        self.last_data_time: Optional[float] = None
        self.connected = False

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # bleak is imported lazily so the rest of the system runs without it.
        try:
            from bleak import BleakScanner, BleakClient
            self._BleakScanner = BleakScanner
            self._BleakClient = BleakClient
            self.available = True
        except Exception as e:
            print(f"[Ultrasonic] bleak not available ({e}); obstacle sensing disabled.")
            self.available = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self):
        """Start the background BLE thread. No-op if bleak is missing."""
        if not self.available or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name='UltrasonicBLE', daemon=True)
        self._thread.start()
        print(f"[Ultrasonic] BLE thread started (looking for '{self.device_name}').")

    def stop(self):
        """Stop the background BLE thread and disconnect."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.scan_timeout + 2.0)
        self._thread = None
        self.connected = False
        print("[Ultrasonic] BLE thread stopped.")

    # ── Public accessors ─────────────────────────────────────────────────

    def get_distances(self) -> Optional[Tuple[float, float]]:
        """Latest (lhs, rhs) distances in cm, or None if nothing received yet."""
        with self._lock:
            if self._lhs is None or self._rhs is None:
                return None
            return self._lhs, self._rhs

    def is_healthy(self) -> bool:
        """True while connected to the sensor and at least one reading has arrived."""
        return self.connected and self.get_distances() is not None

    def is_obstacle(self) -> bool:
        """True if either sensor reads closer than obstacle_distance. False when unhealthy."""
        d = self.get_distances()
        if not self.connected or d is None:
            return False
        return d[0] < self.obstacle_distance or d[1] < self.obstacle_distance

    def filter_direction(self, direction: str) -> str:
        """Return 'stop' instead of `direction` if it is blocked by an obstacle."""
        if direction in self.block_directions and self.is_obstacle():
            return 'stop'
        return direction

    # ── Background BLE loop ──────────────────────────────────────────────

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ble_main())
        finally:
            self._loop.close()
            self._loop = None

    def _on_notify(self, _sender, data: bytearray):
        try:
            lhs, rhs = map(float, data.decode().split(','))
        except Exception as e:
            print(f"[Ultrasonic] Bad packet {data!r}: {e}")
            return
        with self._lock:
            self._lhs, self._rhs = lhs, rhs
            self.last_data_time = time.time()

    async def _ble_main(self):
        # Keep scanning/reconnecting until stop() is called so the sensor can
        # be powered on after the main loop has started.
        while not self._stop_event.is_set():
            address = await self._find_device()
            if address is None:
                await self._sleep(self.reconnect_interval)
                continue
            try:
                async with self._BleakClient(address) as client:
                    await client.start_notify(self.char_uuid, self._on_notify)
                    self.connected = True
                    print(f"[Ultrasonic] Connected to {self.device_name} at {address}.")
                    while not self._stop_event.is_set() and client.is_connected:
                        await asyncio.sleep(0.2)
                    if client.is_connected:
                        await client.stop_notify(self.char_uuid)
            except Exception as e:
                print(f"[Ultrasonic] Connection error: {e}")
            finally:
                self.connected = False
            if not self._stop_event.is_set():
                print("[Ultrasonic] Disconnected — retrying...")
                await self._sleep(self.reconnect_interval)

    async def _find_device(self) -> Optional[str]:
        try:
            devices = await self._BleakScanner.discover(timeout=self.scan_timeout)
        except Exception as e:
            print(f"[Ultrasonic] Scan failed: {e}")
            return None
        for d in devices:
            if d.name == self.device_name:
                return d.address
        return None

    async def _sleep(self, seconds: float):
        """Sleep that returns early when stop() is called."""
        end = time.time() + seconds
        while time.time() < end and not self._stop_event.is_set():
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    # Standalone check: prints distances until Ctrl+C.
    rx = Ultrasonic_Receiver({})
    rx.start()
    try:
        while True:
            d = rx.get_distances()
            state = 'OBSTACLE' if rx.is_obstacle() else 'clear'
            print(f"connected={rx.connected} distances={d} {state}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        rx.stop()
