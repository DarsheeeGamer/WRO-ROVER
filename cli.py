#!/usr/bin/env python3
"""
CLI for Arduino Mega Motor Control
Send commands: F, B, L, R, G, I, H, J, X, x, S
Receive logs from Mega in real-time
"""

import serial
import sys
import threading
import time

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/ttyUSB0"  # Change if different
BAUD_RATE = 9600
# ---------------------------------------

# Open serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
except serial.SerialException as e:
    print(f"Error opening {SERIAL_PORT}: {e}")
    sys.exit(1)

# Thread to continuously read from Mega
def read_from_mega():
    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"[Mega] {line}")
        except Exception as e:
            print(f"[Error reading serial]: {e}")
            break

# Start the read thread
thread = threading.Thread(target=read_from_mega, daemon=True)
thread.start()

# CLI loop
print("=== Arduino Mega CLI ===")
print("Commands: F,B,L,R,G,I,H,J,X,x,S")
print("Type 'Q' to quit")

while True:
    cmd = input(">> ").strip()
    if not cmd:
        continue
    if cmd.upper() == 'Q':
        print("Exiting...")
        break
    if len(cmd) > 1:
        print("Send one character at a time!")
        continue

    try:
        ser.write(cmd.encode('utf-8'))
        # Optional: add a short delay for Arduino processing
        time.sleep(0.05)
    except Exception as e:
        print(f"Error sending command: {e}")

ser.close()
