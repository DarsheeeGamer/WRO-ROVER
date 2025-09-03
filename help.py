# robot_brain.py
# This script runs on the Raspberry Pi 4 to control a robot using Gemini and GPIO serial.

import os
import time
import io
import cv2  # Import OpenCV for camera access
from PIL import Image
import lgpio # Library for direct GPIO access
import enum # For defining an Enum for actions

# Updated import for Google Generative AI
from google import genai
from google.genai import types

# Import Pydantic for structured output schema
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. HARDWARE AND API CONFIGURATION ---

# GPIO Pin Definitions for Software Serial Communication
# These are the pins that will be used to send data to the Arduino.
# Raspberry Pi GPIO 17 (TX) -> Arduino Pin 9 (RX)
# Raspberry Pi GPIO 27 (RX) -> Arduino Pin 10 (TX) (This RX pin is not actively used by this script for receiving from Arduino)
TX_PIN = 17  # Physical GPIO pin number
RX_PIN = 27  # Physical GPIO pin number
BAUD_RATE = 9600
# Calculate the time it takes to transmit one bit at the specified baud rate.
# This is important for accurate software serial communication.
BIT_TIME = 1.0 / BAUD_RATE

# Global variable for the GPIO chip handle, initialized to None.
gpio_handle = None

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
    # Clean up GPIO if camera fails to initialize
    close_gpio()
    exit()
else:
    # Get and print the actual resolution to confirm.
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera initialized using OpenCV. Attempted resolution: 1280x720. Actual resolution: {actual_width}x{actual_height}")
    # Give the camera a moment to warm up and stabilize.
    time.sleep(2)


# --- GPIO Initialization and Communication Functions ---

def init_gpio():
    """Initializes the GPIO chip and claims the necessary pins for serial communication."""
    global gpio_handle
    if gpio_handle is None: # Only initialize if not already done.
        try:
            # Open the GPIO chip. On most Raspberry Pis, this is chip 0.
            gpio_handle = lgpio.gpiochip_open(0)
            # Claim the TX pin as an output pin.
            lgpio.gpio_claim_output(gpio_handle, TX_PIN)
            # Claim the RX pin as an input pin.
            lgpio.gpio_claim_input(gpio_handle, RX_PIN)
            # Set the TX pin to its idle state, which is HIGH for serial communication.
            lgpio.gpio_write(gpio_handle, TX_PIN, 1)
            print(f"GPIO initialized. TX (GPIO {TX_PIN}) as output, RX (GPIO {RX_PIN}) as input.")
        except Exception as e:
            print(f"Error initializing GPIO: {e}")
            print("Ensure 'lgpio' library is installed and you have the necessary permissions.")
            print("You might need to run this script with 'sudo python your_script.py'")
            exit()

def close_gpio():
    """Cleans up GPIO resources by closing the chip and ensuring pins are in a safe state."""
    global gpio_handle
    if gpio_handle:
        try:
            # Ensure the TX pin is in a safe state (HIGH) before closing.
            lgpio.gpio_write(gpio_handle, TX_PIN, 1)
            lgpio.gpiochip_close(gpio_handle)
            print("GPIO resources closed.")
        except Exception as e:
            print(f"Error closing GPIO resources: {e}")
        finally:
            gpio_handle = None # Reset the handle to None.

def send_byte_software_serial(data_byte: int):
    """
    Sends a single byte (an integer from 0-255) to the Arduino using software serial protocol
    over the configured TX_PIN. This function implements the timing for start bit, data bits, and stop bit.
    """
    global gpio_handle
    if gpio_handle is None:
        print("GPIO not initialized. Cannot send byte.")
        return

    # --- Send Start Bit (LOW state for BIT_TIME duration) ---
    lgpio.gpio_write(gpio_handle, TX_PIN, 0)
    time.sleep(BIT_TIME)

    # --- Send Data Bits (LSB first) ---
    # Iterate through each bit of the byte.
    for i in range(8):
        # Extract the i-th bit (0 or 1).
        bit = (data_byte >> i) & 1
        # Set the TX pin to the value of the bit.
        lgpio.gpio_write(gpio_handle, TX_PIN, bit)
        # Wait for the duration of one bit to ensure data integrity.
        time.sleep(BIT_TIME)

    # --- Send Stop Bit (HIGH state for BIT_TIME duration) ---
    lgpio.gpio_write(gpio_handle, TX_PIN, 1)
    time.sleep(BIT_TIME)

def send_command_to_arduino(command: str):
    """
    Sends a single character command to the Arduino by converting it to its ASCII value
    and transmitting it using the software serial protocol.
    """
    if not command or len(command) != 1:
        print(f"Warning: Invalid command '{command}'. Expected a single character.")
        return
   
    # Convert the character command to its ASCII integer representation.
    ascii_value = ord(command)
    send_byte_software_serial(ascii_value)
   
    # Add a small delay after sending the command. This helps ensure the Arduino has
    # time to process the command and stop any previous movement before the next command is sent.
    # This is crucial for sequential movements where the Arduino might still be running
    # a previous command when the next command is transmitted.
    time.sleep(0.05) # A short delay is usually sufficient for communication

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
    init_gpio() # Ensure GPIO is initialized before starting the loop.

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
        # Ensure all resources (camera, GPIO) are cleaned up when the program exits.
        if camera and camera.isOpened(): # Simplified check
            camera.release() # Release the camera resource
        close_gpio()
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
