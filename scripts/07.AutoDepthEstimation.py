import cv2
import numpy as np
import time
from estimate_depth import find_chessboard_depth

def auto_depth_estimation(left_camera_index, right_camera_index, left_calib_file, right_calib_file, stereo_calib_file):
    print(f"Attempting to open left camera with index: {left_camera_index}")
    capL = cv2.VideoCapture(left_camera_index)
    print(f"Attempting to open right camera with index: {right_camera_index}")
    capR = cv2.VideoCapture(right_camera_index)

    if not capL.isOpened():
        print(f"Error: Could not open left camera (index {left_camera_index}). Please check if the camera is connected and not in use. Exiting.")
        return
    else:
        print(f"Successfully opened left camera (index {left_camera_index}).")

    if not capR.isOpened():
        print(f"Error: Could not open right camera (index {right_camera_index}). Please check if the camera is connected and not in use. Exiting.")
        return
    else:
        print(f"Successfully opened right camera (index {right_camera_index}).")

    print("Initializing cameras...")
    # Allow cameras to warm up
    for _ in range(10):
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("Warning: Could not read frame during warm-up. Cameras might not be ready.")
        time.sleep(0.1)

    print("Cameras initialized. Press 'q' to quit.")

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL or not retR:
            print("Error: Failed to capture frames from one or both cameras. Exiting loop.")
            break

        # Estimate depth using the captured frames
        depth = find_chessboard_depth(frameL, frameR, None, None, None)

        if depth is not None:
            print(f"Average Chessboard Depth: {depth:.2f} units")
        else:
            print("Chessboard not found or depth estimation failed.")

        # Display the frames (optional, for debugging/visualization)
        cv2.imshow('Left Camera', frameL)
        cv2.imshow('Right Camera', frameR)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Continuously estimate depth using stereo cameras.')
    parser.add_argument('--left_cam_idx', type=int, default=0, help='Index of the left camera.')
    parser.add_argument('--right_cam_idx', type=int, default=1, help='Index of the right camera.')
    parser.add_argument('--left_calib', type=str, default='calib_left.npz', help='Path to the left camera calibration file.')
    parser.add_argument('--right_calib', type=str, default='calib_right.npz', help='Path to the right camera calibration file.')
    parser.add_argument('--stereo_calib', type=str, default='calibration/stereo_full.npz', help='Path to the stereo calibration file.')
    args = parser.parse_args()

    auto_depth_estimation(args.left_cam_idx, args.right_cam_idx, args.left_calib, args.right_calib, args.stereo_calib)