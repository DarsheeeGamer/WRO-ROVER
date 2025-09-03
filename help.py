# robot_brain.py
# This script runs on the Raspberry Pi 4 to control a robot using Gemini and USB serial.

import io
import os
import time
from typing import Optional

import enum

import cv2
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import comms

# --- 1. HARDWARE AND API CONFIGURATION ---


# Default duration for movement actions if Gemini doesn't specify one or provides an invalid one.
DEFAULT_MOVEMENT_DURATION = 3 # seconds

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

# --- Camera Initialization ---
# Use OpenCV to capture frames from the camera. '0' typically refers to the default camera.
camera = cv2.VideoCapture(0) # Use cam0

# Set camera properties to 720p (1280x720) resolution.
# Note: Not all cameras or Raspberry Pi configurations may support 720p directly.
# If you encounter issues, you might need to adjust these values or the camera module.
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Set frame width to 1280
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Set frame height to 720

if not camera.isOpened():
    print("Error: Could not open camera.")
    # Clean up serial if camera fails to initialise
    comms.close_serial()
    exit()
else:
    # Get and print the actual resolution to confirm.
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera initialized using OpenCV. Attempted resolution: 1280x720. Actual resolution: {actual_width}x{actual_height}")
    # Give the camera a moment to warm up and stabilize.
    time.sleep(2)


# --- GPIO no longer used ---

def send_command_to_arduino(command: str):
    """Forward a one-byte command to the Arduino over USB serial."""
    if not command or len(command) != 1:
        print(f"Warning: Invalid command '{command}'. Expected a single character.")
        return

    comms.send_command(command)
    time.sleep(0.05)  # Brief pause to let Arduino act

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
    global camera # Declare intent to use and modify the global 'camera' variable
    comms.init_serial()  # Open USB serial

    try:
        while True:
            print("\n--- Starting new cycle ---")

            # --- Capture Image from Camera using OpenCV ---
            ret, frame = camera.read() # Read a frame from the camera
            if not ret:
                print("Error: Failed to capture frame from camera.")
                # Attempt to restart camera if reading fails
                camera.release()
                time.sleep(2) # Wait before retrying
                camera = cv2.VideoCapture(0) # Now this re-assigns to the global 'camera'
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Set width to 720p
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # Set height to 720p
                if not camera.isOpened():
                    print("Error: Failed to re-initialize camera. Exiting.")
                    break # Exit loop if camera cannot be re-initialized
                continue # Skip to next iteration if frame capture failed but camera re-initialized

            # Convert the OpenCV frame (NumPy array) to a PIL Image, then to bytes.
            # OpenCV uses BGR color format, PIL uses RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
           
            image_bytes_io = io.BytesIO()
            img_pil.save(image_bytes_io, format='jpeg')
            image_bytes = image_bytes_io.getvalue()
            print("Image captured successfully.")

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
        if camera and camera.isOpened(): # Simplified check
            camera.release() # Release the camera resource
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
