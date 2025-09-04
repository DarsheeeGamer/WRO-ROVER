"""calibrate_arm.py
This script provides a comprehensive solution for stereo camera calibration using a checkerboard pattern.
It is specifically designed for calibrating the mini cameras intended for a robotic arm.

The calibration process involves:
1. Capturing image pairs from the left and right cameras.
2. Detecting checkerboard corners in the captured images.
3. Performing individual camera calibrations for intrinsic parameters.
4. Performing stereo calibration to determine the extrinsic relationship between the two cameras.
5. Saving the calibration results to a NumPy `.npz` file for later use in 3D reconstruction and distance measurement.

Before running, ensure you have:
- OpenCV (`opencv-python`) and NumPy installed (`pip install opencv-python numpy`).
- A physical checkerboard with known dimensions.
- Two cameras connected and accessible (typically at indices 0 and 1).

Adjust `CHECKERBOARD` and `SQUARE_SIZE` variables according to your checkerboard.
"""

import numpy as np
import glob
import os
import cv2

# --- Configuration ---
# The dimensions of your checkerboard (number of inner corners)
CHECKERBOARD = (6, 9)
# The size of a square on your checkerboard (in any unit, e.g., cm, mm, inches)
SQUARE_SIZE = 2.5 # cm

# Number of image pairs to capture for calibration
NUM_IMAGES_TO_CAPTURE = 25

# --- End Configuration ---

def calibrate_stereo_cameras():
    """Performs stereo camera calibration using a checkerboard pattern.

    This function guides the user through capturing image pairs, detecting checkerboard
    corners, and then performing both individual camera calibrations and stereo calibration.
    The results are saved to `stereo_cal_arm.npz`.

    Requires: OpenCV (`cv2`) and NumPy (`np`).
    """
    print("Starting stereo camera calibration...")

    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane for the left camera.
    imgpoints_right = []  # 2d points in image plane for the right camera.

    # Initialize cameras
    print("Initializing cameras...")
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("Error: Could not open one or both cameras.")
        return

    print("Cameras initialized successfully.")
    print(f"Press 'c' to capture an image pair. You need {NUM_IMAGES_TO_CAPTURE} pairs.")
    print("Press 'q' to quit and start calibration (if enough images are captured).")

    captured_images = 0
    while captured_images < NUM_IMAGES_TO_CAPTURE:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Error: Could not read frame from one or both cameras.")
            break

        # Display the frames
        cv2.imshow('Left Camera', frame_left)
        cv2.imshow('Right Camera', frame_right)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print(f"Capturing image pair {captured_images + 1}/{NUM_IMAGES_TO_CAPTURE}...")
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_left_corners, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
            ret_right_corners, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

            # If found, add object points, image points (after refining them)
            if ret_left_corners and ret_right_corners:
                objpoints.append(objp)

                corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                imgpoints_left.append(corners2_left)

                corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
                imgpoints_right.append(corners2_right)

                # Draw and display the corners
                cv2.drawChessboardCorners(frame_left, CHECKERBOARD, corners2_left, ret_left_corners)
                cv2.imshow('Left Camera', frame_left)
                cv2.drawChessboardCorners(frame_right, CHECKERBOARD, corners2_right, ret_right_corners)
                cv2.imshow('Right Camera', frame_right)
                cv2.waitKey(500) # Display for a short time

                captured_images += 1
                print("Image pair captured successfully.")
            else:
                print("Checkerboard not found in one or both images. Please try again.")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("Not enough image pairs captured for calibration. Exiting.")
        return

    print("Starting calibration process...")

    # Calibrate left camera
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    if not ret_left:
        print("Left camera calibration failed.")
        return
    print("Left camera calibrated.")

    # Calibrate right camera
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
    if not ret_right:
        print("Right camera calibration failed.")
        return
    print("Right camera calibrated.")

    # Stereo calibration
    print("Performing stereo calibration...")
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # It is often better to fix the intrinsic parameters and only compute the extrinsic parameters
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_ZERO_TANGENT_DIST

    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    ret_stereo, new_mtx_left, new_dist_left, new_mtx_right, new_dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        gray_left.shape[::-1],
        criteria=stereocalib_criteria,
        flags=flags
    )

    if not ret_stereo:
        print("Stereo calibration failed.")
        return

    print("Stereo calibration successful.")

    # Save the stereo calibration result
    np.savez('stereo_cal_arm.npz',
             mtx_left=new_mtx_left, dist_left=new_dist_left,
             mtx_right=new_mtx_right, dist_right=new_dist_right,
             R=R, T=T, E=E, F=F)

    print("Calibration results saved to 'stereo_cal_arm.npz'")
    print("Calibration complete.")

if __name__ == '__main__':
    calibrate_stereo_cameras()