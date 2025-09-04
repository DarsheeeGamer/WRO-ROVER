import numpy as np
import cv2
import os
import time
import numpy as np
import argparse

import cv2
import numpy as np

def find_chessboard_depth(left_image, right_image, left_calib_file, right_calib_file, stereo_calib_file):
    """
    Finds the average depth of a chessboard in a stereo image pair.

    Args:
        left_image: Left image frame (numpy array).
        right_image: Right image frame (numpy array).
        left_calib_file: Path to the left camera calibration file.
        right_calib_file: Path to the right camera calibration file.
        stereo_calib_file: Path to the stereo calibration file.
    Returns:
        float: The average depth of the chessboard, or None if not found or an error occurs.
    """
    # Step 1: Load stereo calibration matrices
    try:
        stereo_calib_data = np.load('calibration/stereo_full.npz')
        cameraMatrixL = stereo_calib_data['mtx_left']
        distL = stereo_calib_data['dist_left']
        cameraMatrixR = stereo_calib_data['mtx_right']
        distR = stereo_calib_data['dist_right']
        R = stereo_calib_data['R']
        T = stereo_calib_data['T']

        print(f"\n--- Stereo Calibration Data Loaded ---")
        print(f"Left Camera Matrix (mtx_left): {cameraMatrixL.shape}")
        print(f"Left Distortion Coeffs (dist_left): {distL.shape}")
        print(f"Right Camera Matrix (mtx_right): {cameraMatrixR.shape}")
        print(f"Right Distortion Coeffs (dist_right): {distR.shape}")
        print(f"Rotation Matrix (R): {R.shape}")
        print(f"Translation Vector (T): {T.shape}")
        print(f"--------------------------------------\n")

    except FileNotFoundError:
        print("Error: stereo_cal.npz not found. Please run auto_calibrate.py first.")
        return None
    except Exception as e:
        print(f"Error loading stereo calibration data: {e}")
        return None
    
    if left_image is None or right_image is None:
        print("Error: Received empty image frames.")
        return None

    h, w, _ = left_image.shape

    # Step 2: Rectify the Images
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrixL, distL, cameraMatrixR, distR, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )
    print(f"Q matrix: {Q}")
    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrixL, distL, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrixR, distR, R2, P2, (w, h), cv2.CV_32FC1)

    rectified_left = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

    # Display rectified images for debugging
    cv2.imshow('Rectified Left (Debug)', rectified_left)
    cv2.imshow('Rectified Right (Debug)', rectified_right)

    # Step 3: Compute Disparity Map
    gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=480, blockSize=31)
    stereo.setMinDisparity(0)
    stereo.setNumDisparities(480) # Should be divisible by 16
    stereo.setBlockSize(31) # Odd number, 5-25 range
    stereo.setDisp12MaxDiff(1)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleWindowSize(200)
    stereo.setSpeckleRange(64)
    stereo.setPreFilterCap(63)
    stereo.setPreFilterSize(15)

    disparity_map = stereo.compute(gray_left, gray_right).astype(np.float32)
    print(f"Disparity map shape: {disparity_map.shape}, min: {np.min(disparity_map)}, max: {np.max(disparity_map)}")
    # Normalize disparity map for visualization and save it
    disp_vis = cv2.normalize(disparity_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite('disparity_map_debug.png', disp_vis)

    # Filter out bad disparity values
    disparity_map[disparity_map == 0] = np.nan

    # Step 4: Find Chessboard Corners on Rectified Images
    chessboardSize = (7,10) # Matches your calibration setup
    retL, cornersL = cv2.findChessboardCorners(gray_left, chessboardSize, None)
    
    if not retL:
        # print("Chessboard not found in the left rectified image.")
        return None

    # Draw and display corners for debugging
    img_corners = rectified_left.copy()
    cv2.drawChessboardCorners(img_corners, chessboardSize, cornersL, retL)
    cv2.imshow('Chessboard Corners (Debug)', img_corners)

    # Step 5: Reproject and Calculate Depth
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    print(f"Points 3D shape: {points_3D.shape}, min Z: {np.nanmin(points_3D[:,:,2])}, max Z: {np.nanmax(points_3D[:,:,2])}")

    # Get the Z-coordinate (depth) for each detected corner.
    depth_values = []
    for corner in cornersL:
        x, y = int(corner[0, 0]), int(corner[0, 1])
        depth = points_3D[y, x, 2] # The Z-coordinate is the depth
        if not np.isinf(depth) and not np.isnan(depth) and depth > 0:
            depth_values.append(depth)
    print(f"Individual depth values: {depth_values}")

    if not depth_values:
        # print("No valid depth values could be calculated for the chessboard corners.")
        return None
    else:
        avg_depth = np.mean(depth_values)
        # avg_depth = avg_depth*(15.000/6.23766697) # Removed hardcoded scaling factor
        return avg_depth

# The main execution block is removed as this file will now be imported as a module.