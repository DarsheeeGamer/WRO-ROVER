"""Lightweight wrapper around :pymod:`pyserial` for Arduino communication.

All functions are no-ops if the port is unopened, preventing crashes during
initialisation failures.
"""

import time
from typing import Optional

import serial

# USB serial device name for the Arduino
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 9600

# Internal handle for the opened serial port
_serial: Optional[serial.Serial] = None


def init_serial(port: str = SERIAL_PORT, baud_rate: int = BAUD_RATE) -> None:
    """Open the serial connection if it is not already open."""
    global _serial
    if _serial is None or not _serial.is_open:
        _serial = serial.Serial(port, baud_rate, timeout=1)
        # Give the port some time to stabilise (recommended for some USB-UART chips)
        time.sleep(2)
        print(f"Serial connection opened on {port} @ {baud_rate} baud.")


def close_serial() -> None:
    """Close the serial connection cleanly."""
    global _serial
    if _serial and _serial.is_open:
        _serial.close()
        print("Serial connection closed.")
    _serial = None


def send_command(cmd: str) -> None:
    """Send a single-character command to the Arduino.

    Args:
        cmd: Exactly one ASCII character.
    """
    if not cmd or len(cmd) != 1:
        print(f"Invalid command '{cmd}'. Must be a single character.")
        return

    if _serial is None or not _serial.is_open:
        print("Serial connection not initialised. Call init_serial() first.")
        return

    _serial.write(cmd.encode("ascii"))
    _serial.flush()
