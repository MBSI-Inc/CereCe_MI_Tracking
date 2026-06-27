"""
Linux RFCOMM socket shim for explorepy's C++ SDK (_exploresdk).

The Mentalab C++ SDK drops the RFCOMM connection ~45ms after streaming starts
on Linux. A raw Python socket stays stable, but conda Python 3.10 was compiled
without AF_BLUETOOTH. This shim uses ctypes to call the C socket() API directly,
bypassing Python's socket module entirely, and does exact reads so the parser
stays byte-aligned.

Applied in eeg_receiver.py on Linux before Explore.connect():
    from explorepy import btcpp as _btcpp
    import modules.explorepy_linux_socket_shim as _shim
    _btcpp.exploresdk = _shim
"""

import ctypes
import ctypes.util
import errno
import logging
import os
import select
import struct
import threading

_log = logging.getLogger(__name__)

# Linux Bluetooth constants
AF_BLUETOOTH = 31
BTPROTO_RFCOMM = 3
SOCK_STREAM = 1
# fcntl
F_GETFL = 3
F_SETFL = 4
O_NONBLOCK = 0o4000
# setsockopt
SOL_SOCKET = 1
SO_ERROR = 4


def _mac_to_bdaddr(mac_str):
    """'AA:BB:CC:DD:EE:FF' → bytes in reverse order (little-endian bdaddr_t)."""
    return bytes(reversed([int(x, 16) for x in mac_str.split(':')]))


class _SockAddrRC(ctypes.Structure):
    """struct sockaddr_rc { sa_family_t rc_family; bdaddr_t rc_bdaddr; uint8_t rc_channel; }"""
    _fields_ = [
        ('rc_family',  ctypes.c_uint16),
        ('rc_bdaddr',  ctypes.c_uint8 * 6),
        ('rc_channel', ctypes.c_uint8),
    ]


class _BTSerialPortBinding:
    def __init__(self, mac_address, channel):
        self._mac = mac_address
        self._channel = channel
        self._fd = -1
        self._libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

    def Connect(self, timeout=8.0):
        fd = -1
        try:
            libc = self._libc
            fd = libc.socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM)
            if fd < 0:
                _log.debug("socket() failed errno=%d", ctypes.get_errno())
                return 1

            # Build sockaddr_rc
            addr = _SockAddrRC()
            addr.rc_family = AF_BLUETOOTH
            bdaddr = _mac_to_bdaddr(self._mac)
            for i, b in enumerate(bdaddr):
                addr.rc_bdaddr[i] = b
            addr.rc_channel = self._channel

            # Set non-blocking so connect() returns immediately with EINPROGRESS
            flags = libc.fcntl(fd, F_GETFL, 0)
            libc.fcntl(fd, F_SETFL, flags | O_NONBLOCK)

            ret = libc.connect(fd, ctypes.byref(addr), ctypes.sizeof(addr))
            err = ctypes.get_errno()

            if ret < 0 and err != errno.EINPROGRESS:
                _log.debug("connect() failed immediately errno=%d", err)
                os.close(fd)
                return 1

            # Wait up to `timeout` seconds for the connection to complete
            _, writable, _ = select.select([], [fd], [], timeout)
            if not writable:
                _log.debug("connect() timed out after %ss", timeout)
                os.close(fd)
                return 1

            # Check SO_ERROR
            so_err = ctypes.c_int(0)
            so_err_len = ctypes.c_uint(ctypes.sizeof(so_err))
            libc.getsockopt(fd, SOL_SOCKET, SO_ERROR,
                            ctypes.byref(so_err), ctypes.byref(so_err_len))
            if so_err.value != 0:
                _log.debug("connect() completed with SO_ERROR=%d", so_err.value)
                os.close(fd)
                return 1

            # Restore blocking mode for reads/writes
            libc.fcntl(fd, F_SETFL, flags)

            self._fd = fd
            return 0

        except Exception as exc:
            _log.debug("Connect exception: %s", exc)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            return 1

    def Read(self, n_bytes):
        """Read exactly n_bytes; return as surrogateescape string (btcpp.py re-encodes)."""
        buf = bytearray()
        while len(buf) < n_bytes:
            chunk = os.read(self._fd, n_bytes - len(buf))
            if not chunk:
                raise IOError("connection has been closed")
            buf.extend(chunk)
        return bytes(buf).decode('utf-8', errors='surrogateescape')

    def Write(self, data):
        os.write(self._fd, bytes(data))

    def Close(self):
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1


class _BTDevice:
    def __init__(self, name, address):
        self.name = name
        self.address = address


class _ExploreSDK:
    def PerformDeviceSearch(self):
        """Stub — MAC is cached in settings so this path is rarely taken."""
        return []


def BTSerialPortBinding_Create(mac_address, channel):
    return _BTSerialPortBinding(mac_address, channel)


def ExploreSDK_Create():
    return _ExploreSDK()
