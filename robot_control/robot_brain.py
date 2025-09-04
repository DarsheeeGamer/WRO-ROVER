"""robot_brain.py
This script serves as the central control unit for the robot, integrating Gemini AI for intelligent decision-making and managing hardware interactions via USB serial communication.

It handles:
- Gemini API configuration and interaction.
- Stereo camera initialization and image acquisition.
- Loading of stereo calibration data for accurate depth perception.
- Computation of real-world distances from stereo camera feeds.
- Sending commands to the Arduino for robot movement and actions.

This script is designed to run on a powerful computing platform (e.g., a laptop running Ubuntu Server) that can handle both AI processing and camera stream processing.
"""

import io
import os
import time
from typing import Optional

import enum

import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import comms
import logging
import builtins

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Route all existing print calls to the module logger for consistency.
builtins.print = lambda *args, **kwargs: logger.info(" ".join(map(str, args)))

# --- 1. HARDWARE AND API CONFIGURATION ---


# Default duration for movement actions if Gemini doesn't specify one or provides an invalid one.
DEFAULT_MOVEMENT_DURATION = 3  # seconds

# --- 2. GEMINI API INITIALIZATION ---

# Configure the Gemini client using the API key
# It's highly recommended to load API keys from environment variables for security.
# For example: API_KEY = os.getenv("GOOGLE_API_KEY")
# For now, it's hardcoded as in your original snippet:

# Configure the Gemini client using the API key
try:
    # It's highly recommended to load API keys from environment variables for security.
    # For example: API_KEY = os.getenv("GOOGLE_API_KEY")
    # For now, it's hardcoded as in your original snippet:
    client = genai.Client(api_key="AIzaSyC6dMTXJKNWOI36MgtPA53ywftHDtXTOow")
    list(client.models.list()) # Test by listing models to ensure the client is working.
    print("Google GenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing Google GenAI client: {e}")
    print("Ensure the 'google-genai' library is installed (`pip install google-genai`)")
    print("and that the GOOGLE_API_KEY environment variable is set.")
    exit()

# --- 3. STEREO CAMERA CALIBRATION AND INITIALIZATION ---

"""
Functions and variables related to loading stereo calibration data and initializing cameras.
"""

# Load stereo calibration data
def _load_stereo_calibration():
    """Loads stereo calibration parameters from various .npz files.

    It attempts to load from 'stereo_full.npz', 'calib_left.npz', and 'calib_right.npz'.
    These files contain camera matrices, distortion coefficients, and rectification parameters
    necessary for stereo vision.

    Returns:
        dict: A dictionary containing all loaded calibration parameters.
    """
    calib = {}
    try:
        sf = np.load('stereo_full.npz', allow_pickle=True)
        for k in sf.files:
            calib[k] = sf[k]
        print("Loaded stereo_full.npz calibration.")
    except Exception as e:
        print(f"Warning: Failed to load stereo_full.npz: {e}")
    for fname, prefix in [("calib_left.npz", "left_"), ("calib_right.npz", "right_")]:
        try:
            f = np.load(fname, allow_pickle=True)
            for k in f.files:
                calib[prefix + k] = f[k]
            print(f"Loaded {fname} calibration.")
        except Exception as e:
            print(f"Warning: Failed to load {fname}: {e}")
    return calib

_stereo_calib = _load_stereo_calibration()

# Helper to get first present key
def _g(d, keys):
    """Helper function to retrieve a value from a dictionary given a list of possible keys.

    Args:
        d (dict): The dictionary to search within.
        keys (list): A list of keys to try, in order of preference.

    Returns:
        Any: The value associated with the first found key, or None if no key is found.
    """
    for k in keys:
        if k in d:
            return d[k]
    return None

# Extract matrices if present
_K1 = _g(_stereo_calib, ["K1", "cameraMatrix1", "left_camera_matrix", "mtx1", "left_mtx"])
_D1 = _g(_stereo_calib, ["D1", "distCoeffs1", "left_dist", "dist1", "left_distortion"])
_K2 = _g(_stereo_calib, ["K2", "cameraMatrix2", "right_camera_matrix", "mtx2", "right_mtx"])
_D2 = _g(_stereo_calib, ["D2", "distCoeffs2", "right_dist", "dist2", "right_distortion"])
_R1 = _g(_stereo_calib, ["R1", "left_R"])
_R2 = _g(_stereo_calib, ["R2", "right_R"])
_P1 = _g(_stereo_calib, ["P1", "left_P"])
_P2 = _g(_stereo_calib, ["P2", "right_P"])
_Q  = _g(_stereo_calib, ["Q"])

# Rectification maps (lazy-built on first frame if matrices exist)
_maps_ready = False
_map1x = _map1y = _map2x = _map2y = None

# Open stereo cameras
# Camera indices (0 and 1) might vary depending on the system and camera connection order.
# Ensure these correspond to your left and right cameras.


# Open stereo cameras
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(1)

for cam in (left_camera, right_camera):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not left_camera.isOpened() or not right_camera.isOpened():
    print("Error: Could not open both stereo cameras (indices 0 and 1).")
    comms.close_serial()
    exit()
else:
    lw = int(left_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    lh = int(left_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rw = int(right_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    rh = int(right_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stereo cameras initialized. Left: {lw}x{lh}, Right: {rw}x{rh}")
    time.sleep(2)

# Stereo matcher (tuned modestly for speed)
# Parameters are set for a balance between accuracy and performance.
_stereo_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*6,  # must be divisible by 16
    blockSize=5,
    P1=8*3*5*5,
    P2=32*3*5*5,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=1,
)

def _ensure_rectification_maps(frame_shape):
    """Ensures that stereo rectification maps are prepared.

    These maps are used to undistort and rectify stereo image pairs,
    making their epipolar lines horizontal and simplifying disparity calculation.
    This function is called lazily on the first frame processing.

    Args:
        frame_shape (tuple): The shape of the input image frames (height, width, channels).
    """
    global _maps_ready, _map1x, _map1y, _map2x, _map2y
    if _maps_ready:
        return
    if _K1 is None or _D1 is None or _R1 is None or _P1 is None or _K2 is None or _D2 is None or _R2 is None or _P2 is None:
        print("Rectification matrices incomplete; proceeding without rectification.")
        _maps_ready = True  # avoid repeated warnings
        return
    h, w = frame_shape[:2]
    P1nm = _P1[:, :3] if _P1 is not None and getattr(_P1, 'shape', (0, 0))[1] == 4 else _P1
    P2nm = _P2[:, :3] if _P2 is not None and getattr(_P2, 'shape', (0, 0))[1] == 4 else _P2
    _map1x, _map1y = cv2.initUndistortRectifyMap(_K1, _D1, _R1, P1nm, (w, h), cv2.CV_32FC1)
    _map2x, _map2y = cv2.initUndistortRectifyMap(_K2, _D2, _R2, P2nm, (w, h), cv2.CV_32FC1)
    _maps_ready = True
    print("Rectification maps prepared.")


# --- 4. ARDUINO COMMUNICATION FUNCTIONS ---

# GPIO no longer used directly in this script; communication is via serial.

def send_command_to_arduino(command: str):
    """Sends a single-character command to the Arduino over USB serial.

    This function acts as an intermediary, using the `comms` module to send
    commands to the connected Arduino, which then executes the corresponding action.

    Args:
        command (str): A single ASCII character representing the command to send.
                       e.g., 'F' for forward, 'B' for backward, 'L' for left, 'R' for right.
    """
    """Forward a one-byte command to the Arduino over USB serial."""
    if not command or len(command) != 1:
        print(f"Warning: Invalid command '{command}'. Expected a single character.")
        return

    comms.send_command(command)
    time.sleep(0.05)  # Brief pause to let Arduino act


def compute_distance_meters(left_bgr, right_bgr) -> Optional[float]:
    """Computes the forward distance to objects using stereo disparity and calibration data.

    This function takes rectified stereo image pairs, computes their disparity map,
    and then reprojects them into 3D space using the loaded stereo calibration parameters.
    The median Z-coordinate (depth) in a central Region of Interest (ROI) is returned
    as the estimated distance.

    Args:
        left_bgr (np.array): The BGR image frame from the left camera.
        right_bgr (np.array): The BGR image frame from the right camera.

    Returns:
        Optional[float]: The estimated distance in meters to the object in the central ROI,
                         or None if the distance cannot be computed (e.g., invalid frames,
                         missing calibration, or no valid disparity).
    """
    """Compute forward distance using stereo disparity and calibration.

    Returns median Z (meters) in a central ROI, or None if unavailable.
    """
    global _Q
    if left_bgr is None or right_bgr is None:
        return None
    _ensure_rectification_maps(left_bgr.shape)
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    if _map1x is not None and _map1y is not None and _map2x is not None and _map2y is not None:
        left_gray = cv2.remap(left_gray, _map1x, _map1y, cv2.INTER_LINEAR)
        right_gray = cv2.remap(right_gray, _map2x, _map2y, cv2.INTER_LINEAR)

    disp = _stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    if _Q is not None:
        points_3d = cv2.reprojectImageTo3D(disp, _Q)
        Z = points_3d[:, :, 2]
    else:
        # fallback using Z = f*B / d from P1,P2 if Q not present
        if _P1 is None or _P2 is None:
            return None
        fx = float(_P1[0, 0])
        B = abs(float(_P2[0, 3]) / fx) if fx != 0 else None
        if B is None or fx == 0:
            return None
        with np.errstate(divide='ignore'):
            Z = (fx * B) / disp

    h, w = disp.shape
    cx0, cx1 = int(w*0.45), int(w*0.55)
    cy0, cy1 = int(h*0.45), int(h*0.55)
    roi = Z[cy0:cy1, cx0:cx1]
    # Filter invalid values
    valid = np.isfinite(roi) & (roi > 0.05) & (roi < 10.0)
    if not np.any(valid):
        return None
    med = float(np.median(roi[valid]))
    return med

# --- 2. DEFINE THE ROBOT'S ACTIONS (TOOLS) ---
# These are Python functions that will be called based on Gemini's structured output.

def move_forward_action(duration: int):
    """
    Moves the robot forward in a straight line for a specified duration and then stops.
    Args:
        duration (int): The duration in seconds to move forward. Must be a positive integer.
    """
    print(f"Executing: Move Forward for {duration} seconds.")
    send_command_to_arduino('F')  # Send command to start moving forward
    time.sleep(duration)          # Wait for the specified duration
    send_command_to_arduino('S')  # Send command to stop
    return {"status": "success", "action": "move_forward", "duration": duration}

def move_backward_action(duration: int):
    """
    Moves the robot backward in a straight line for a specified duration and then stops.
    Args:
        duration (int): The duration in seconds to move backward. Must be a positive integer.
    """
    print(f"Executing: Move Backward for {duration} seconds.")
    send_command_to_arduino('B')  # Send command to start moving backward
    time.sleep(duration)          # Wait for the specified duration
    send_command_to_arduino('S')  # Send command to stop
    return {"status": "success", "action": "move_backward", "duration": duration}

def turn_left_action(duration: int):
    """
    Rotates the robot to the left on the spot for a specified duration and then stops.
    Args:
        duration (int): The duration in seconds to turn left. Must be a positive integer.
    """
    print(f"Executing: Turn Left for {duration} seconds.")
    send_command_to_arduino('L')  # Send command to start turning left
    time.sleep(duration)          # Wait for the specified duration
    send_command_to_arduino('S')  # Send command to stop
    return {"status": "success", "action": "turn_left", "duration": duration}

def turn_right_action(duration: int):
    """
    Rotates the robot to the right on the spot for a specified duration and then stops.
    Args:
        duration (int): The duration in seconds to turn right. Must be a positive integer.
    """
    print(f"Executing: Turn Right for {duration} seconds.")
    send_command_to_arduino('R')  # Send command to start turning right
    time.sleep(duration)          # Wait for the specified duration
    send_command_to_arduino('S')  # Send command to stop
    return {"status": "success", "action": "turn_right", "duration": duration}

def stop_moving_action():
    """
    Stops any current movement of the robot.
    """
    print("Executing: Stop Moving.")
    send_command_to_arduino('S')  # Send command to stop
    return {"status": "success", "action": "stop_moving"}

# A dictionary to map string action names from the Pydantic model to their corresponding Python functions.
AVAILABLE_ACTIONS = {
    "move_forward": move_forward_action,
    "move_backward": move_backward_action,
    "turn_left": turn_left_action,
    "turn_right": turn_right_action,
    "stop_moving": stop_moving_action,
}

# --- 3. DEFINE PYDANTIC MODEL FOR STRUCTURED OUTPUT ---
# This model describes the JSON structure we expect from Gemini.

class RobotActionEnum(str, enum.Enum):
    """Enumeration of possible robot actions."""
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP_MOVING = "stop_moving"

class RobotCommand(BaseModel):
    """
    Schema for the robot's command, including an action and an optional duration.
    """
    action: RobotActionEnum = Field(description="The action the robot should take.")
    duration: Optional[int] = Field(
        None,
        ge=1, # Duration must be greater than or equal to 1 second
        description="The duration in seconds for movement actions. Only applicable for move_forward, move_backward, turn_left, and turn_right. Must be a positive integer."
    )

# --- 4. INITIALIZE THE GEMINI MODEL (No change to model name) ---
client_model_name = "gemini-2.5-flash-lite"

# --- 5. MAIN EXECUTION LOOP ---

def main():
    """
    The main loop that orchestrates the robot's operation:
    1. Captures an image from the camera.
    2. Sends the image to Gemini for analysis.
    3. Parses Gemini's structured output to determine the next action.
    4. Executes the action by sending commands to the Arduino.
    5. Repeats.
    """
    global left_camera, right_camera  # Access and potentially rebind stereo cameras
    comms.init_serial()  # Open USB serial

    try:
        while True:
            print("\n--- Starting new cycle ---")

            # --- Capture Images from Stereo Cameras ---
            retL, left_frame = left_camera.read()
            retR, right_frame = right_camera.read()
            if not retL or not retR:
                print("Error: Failed to capture frame from one or both cameras.")
                # Attempt to restart cameras
                try:
                    left_camera.release(); right_camera.release()
                except Exception:
                    pass
                time.sleep(2)
                left_camera = cv2.VideoCapture(0)
                right_camera = cv2.VideoCapture(1)
                for cam in (left_camera, right_camera):
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if not left_camera.isOpened() or not right_camera.isOpened():
                    print("Error: Failed to re-initialize stereo cameras. Exiting.")
                    break
                continue

            # Convert the OpenCV frame (NumPy array) to a PIL Image, then to bytes.
            # OpenCV uses BGR color format, PIL uses RGB.
            frame_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
           
            image_bytes_io = io.BytesIO()
            img_pil.save(image_bytes_io, format='jpeg')
            image_bytes = image_bytes_io.getvalue()
            print("Image captured successfully.")

            # Compute distance using stereo
            distance_m = compute_distance_meters(left_frame, right_frame)
            if distance_m is not None:
                print(f"Estimated forward distance: {distance_m:.2f} m")
            else:
                print("Estimated forward distance: unavailable")

            # Prepare the captured image data in a format suitable for the Gemini API.
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg'
            )

            # --- Analyze Image with Gemini and Get Action Recommendation ---
            print("Sending image to Gemini for analysis and action planning...")
           
            # The prompt now guides Gemini to fill the JSON structure
            prompt_text = (
                "Analyze this image from my point of view and decide the next action to take. "
                "You MUST respond with a JSON object containing an 'action' field and, if "
                "it's a movement action, a 'duration' field. "
                "Possible actions are 'move_forward', 'move_backward', 'turn_left', 'turn_right', or 'stop_moving'. "
                "For movement actions, the duration should be a positive integer in seconds (e.g., 3). "
                "If the path ahead is clear, choose 'move_forward'. "
                "If there is an obstacle, decide whether to 'turn_left' or 'turn_right' to avoid it. "
                "If the path is blocked, the image is too unclear to make a confident movement, "
                "or no movement is necessary, choose 'stop_moving'. "
                "ALWAYS provide a 'duration' for movement actions (move_forward, move_backward, turn_left, turn_right)."
            )
            if distance_m is not None:
                prompt_text += f" The estimated forward distance is about {distance_m:.2f} meters; consider this when choosing the action."
           
            response = client.models.generate_content(
                model=client_model_name,
                contents=[image_part, prompt_text],
                config=types.GenerateContentConfig(
                    # --- CONFIGURE STRUCTURED OUTPUT HERE ---
                    response_mime_type="application/json",
                    response_schema=RobotCommand, # Use our Pydantic model for schema
                    system_instruction=(
                        "You are a helpful robot assistant. Your goal is to navigate a room based on camera input. "
                        "You *must* respond with a JSON object that strictly adheres to the provided schema: "
                        "{'action': 'RobotActionEnum', 'duration': Optional[int]}. "
                        "Always analyze the image and select the most appropriate single action from "
                        "['move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop_moving']. "
                        "For any action other than 'stop_moving', you *must* include a 'duration' in seconds. "
                        "If the path is clear, prioritize moving forward. If blocked, prioritize turning. "
                        "If unable to move or path is unclear, always choose 'stop_moving'."
                    )
                )
            )
           
            # --- PRINT RAW GEMINI RESPONSE ---
            print("\n--- RAW GEMINI RESPONSE ---")
            print(response)
            print("---------------------------\n")

            # --- Process Gemini's Structured Response and Execute the Determined Action ---
            try:
                robot_command: Optional[RobotCommand] = None

                # Access the parsed Pydantic object directly
                if response.parsed:
                    robot_command = response.parsed
                else:
                    print(f"Warning: Gemini response.parsed is empty or invalid.")
                    # Attempt to parse raw text as JSON as a fallback
                    try:
                        raw_json = json.loads(response.text)
                        robot_command = RobotCommand(**raw_json)
                        print("Successfully parsed raw text as RobotCommand.")
                    except (json.JSONDecodeError, ValidationError) as parse_error:
                        print(f"Error parsing raw text as JSON: {parse_error}")
                        print(f"Raw response text that failed parsing: {response.text}")
                        print("Falling back to stop_moving due to unparseable response.")
                        robot_command = RobotCommand(action=RobotActionEnum.STOP_MOVING)

                if robot_command:
                    action_name = robot_command.action.value
                    print(f"Gemini recommended action: '{action_name}'")

                    if action_name in AVAILABLE_ACTIONS:
                        action_function = AVAILABLE_ACTIONS[action_name]
                       
                        if action_name == RobotActionEnum.STOP_MOVING.value:
                            result = action_function()
                        else: # Movement actions
                            duration = robot_command.duration
                            if not isinstance(duration, int) or duration <= 0:
                                print(f"Warning: Invalid or missing duration '{duration}' for '{action_name}'. Using default {DEFAULT_MOVEMENT_DURATION}s.")
                                duration = DEFAULT_MOVEMENT_DURATION
                            result = action_function(duration=duration)
                       
                        print(f"Action result: {result}")
                    else:
                        print(f"Error: Gemini recommended an unknown action '{action_name}'. Stopping for safety.")
                        result = stop_moving_action()
                        print(f"Action result: {result}")

                else:
                    # Should be covered by the parsing fallback, but as a final safety net
                    print("Gemini provided no interpretable command. Stopping as a precaution.")
                    result = stop_moving_action()
                    print(f"Action result: {result}")

            except Exception as e: # Catch any other unexpected errors during processing
                print(f"Error processing Gemini's structured response: {e}")
                print(f"Raw Gemini response text: {response.text}") # Print raw text for debugging
                print("Stopping the robot due to a processing error.")
                result = stop_moving_action()
                print(f"Action result: {result}")

            # Wait for a short period before the next cycle to allow actions to complete
            time.sleep(3) # Wait 3 seconds before capturing the next image.

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        # Ensure all resources (camera, serial) are cleaned up when the program exits.
        if left_camera and left_camera.isOpened():
            left_camera.release()
        if right_camera and right_camera.isOpened():
            right_camera.release()
        comms.close_serial()
        print("Robot program finished. Resources cleaned up.")

if __name__ == "__main__":
    # Add a check for pydantic if you haven't installed it
    try:
        import pydantic
    except ImportError:
        print("Pydantic library not found. Please install it: pip install pydantic")
        exit()
   
    # Import json for raw text parsing fallback and ValidationError for pydantic
    import json
    from pydantic import ValidationError

    main()
