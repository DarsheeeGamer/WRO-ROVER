"""comms.py
This module provides a lightweight and robust wrapper around the `pyserial` library,
facilitating reliable serial communication with an Arduino or similar microcontroller.

It handles:
- Initializing and closing the serial connection.
- Sending single-character commands to the connected device.
- Graceful handling of unopened ports to prevent crashes.

This module is designed to be used by the main robot control script (`robot_brain.py`)
to abstract away the complexities of direct serial port management.
"""

import logging
import time
from typing import Optional
import os

import serial

logger = logging.getLogger(__name__)

# Determine a sensible default serial port, with environment variable overrides.
def _detect_default_port() -> str:
    # Environment variables take precedence
    env = os.environ.get("ROBOT_SERIAL_PORT") or os.environ.get("SERIAL_PORT")
    if env:
        return env
    # Platform-based defaults
    if os.name == "nt":
        return "COM3"  # Common Arduino default on Windows
    # POSIX: prefer /dev/ttyUSB0 if present, else fall back to /dev/ttyACM0
    try:
        if os.path.exists("/dev/ttyUSB0"):
            return "/dev/ttyUSB0"
    except Exception:
        pass
    return "/dev/ttyACM0"

SERIAL_PORT = _detect_default_port()
BAUD_RATE = 9600

# Internal handle for the opened serial port
_serial: Optional[serial.Serial] = None


def init_serial(port: str = SERIAL_PORT, baud_rate: int = BAUD_RATE) -> None:
    """Initializes and opens the serial connection to the Arduino.

    If the serial port is already open, this function does nothing.
    It also includes a brief delay after opening to allow the port to stabilize,
    which is often recommended for USB-to-serial converters.

    Args:
        port (str): The serial port to connect to.
            Detection order:
            1) ROBOT_SERIAL_PORT or SERIAL_PORT env var (if set)
            2) Windows: COM3; POSIX: /dev/ttyUSB0 if present, else /dev/ttyACM0
        baud_rate (int): The baud rate for the serial communication (e.g., 9600).
    """
    global _serial
    if _serial is None or not _serial.is_open:
        _serial = serial.Serial(port, baud_rate, timeout=1)
        # Give the port some time to stabilise (recommended for some USB-UART chips)
        time.sleep(2)
        logger.info("Serial connection opened on %s @ %s baud.", port, baud_rate)


def close_serial() -> None:
    """Closes the serial connection cleanly.

    If the serial port is open, it will be closed. If it's already closed or was never
    initialized, this function does nothing.
    """
    global _serial
    if _serial and _serial.is_open:
        _serial.close()
        logger.info("Serial connection closed.")
    _serial = None


def send_command(cmd: str) -> None:
    """Sends a single-character command to the connected Arduino.

    This function ensures that the command is a single ASCII character and that
    the serial connection is active before attempting to write.

    Args:
        cmd (str): The single ASCII character command to send.
                   If the command is not a single character, a warning is logged.
    """
    if not cmd or len(cmd) != 1:
        logger.warning("Invalid command '%s'. Must be a single character.", cmd)
        return

    if _serial is None or not _serial.is_open:
        logger.warning("Serial connection not initialised. Call init_serial() first.")
        return

    _serial.write(cmd.encode("ascii"))
    _serial.flush()
