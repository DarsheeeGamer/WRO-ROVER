import cv2
import numpy as np
import glob
import os

# --- Configuration ---
# The dimensions of your checkerboard (number of inner corners)
CHECKERBOARD = (6, 9)
# The size of a square on your checkerboard (in any unit, e.g., cm, mm, inches)
SQUARE_SIZE = 2.5 # cm

# Number of image pairs to capture for calibration
NUM_IMAGES_TO_CAPTURE = 25

# --- End Configuration ---

def calibrate_stereo_cameras():
    """
    Performs stereo camera calibration using a checkerboard pattern.
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
        return None # Return None to indicate failure

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
        return None

    print("Starting calibration process...")

    # Calibrate left camera
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    if not ret_left:
        print("Left camera calibration failed.")
        return None
    print("Left camera calibrated.")

    # Calibrate right camera
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
    if not ret_right:
        print("Right camera calibration failed.")
        return None
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
        return None

    print("Stereo calibration successful.")

    # Return the calibration results instead of saving to file
    return {
        'mtx_left': new_mtx_left, 'dist_left': new_dist_left,
        'mtx_right': new_mtx_right, 'dist_right': new_dist_right,
        'R': R, 'T': T, 'E': E, 'F': F
    }

def find_chessboard_depth(left_image, right_image, calib_data):
    """
    Finds the average depth of a chessboard in a stereo image pair.

    Args:
        left_image: Left image frame (numpy array).
        right_image: Right image frame (numpy array).
        calib_data: Dictionary containing calibration matrices (mtx_left, dist_left, mtx_right, dist_right, R, T).
    Returns:
        float: The average depth of the chessboard, or None if not found or an error occurs.
    """
    try:
        cameraMatrixL = calib_data['mtx_left']
        distL = calib_data['dist_left']
        cameraMatrixR = calib_data['mtx_right']
        distR = calib_data['dist_right']
        R = calib_data['R']
        T = calib_data['T']

        print(f"\n--- Stereo Calibration Data Loaded ---")
        print(f"Left Camera Matrix (mtx_left): {cameraMatrixL.shape}")
        print(f"Left Distortion Coeffs (dist_left): {distL.shape}")
        print(f"Right Camera Matrix (mtx_right): {cameraMatrixR.shape}")
        print(f"Right Distortion Coeffs (dist_right): {distR.shape}")
        print(f"Rotation Matrix (R): {R.shape}")
        print(f"Translation Vector (T): {T.shape}")
        print(f"--------------------------------------\n")

    except KeyError as e:
        print(f"Error: Missing calibration data key: {e}")
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

if __name__ == '__main__':
    # Attempt to load calibration data first
    calib_file_path = 'calibration/stereo_full.npz'
    calib_data = None
    if os.path.exists(calib_file_path):
        try:
            loaded_data = np.load(calib_file_path)
            calib_data = {
                'mtx_left': loaded_data['mtx_left'], 'dist_left': loaded_data['dist_left'],
                'mtx_right': loaded_data['mtx_right'], 'dist_right': loaded_data['dist_right'],
                'R': loaded_data['R'], 'T': loaded_data['T'], 'E': loaded_data['E'], 'F': loaded_data['F']
            }
            print("Loaded existing calibration data.")
        except Exception as e:
            print(f"Error loading calibration data from {calib_file_path}: {e}")
            # If loading fails, delete the corrupted file to force recalibration
            if os.path.exists(calib_file_path):
                os.remove(calib_file_path)
                print(f"Deleted corrupted calibration file: {calib_file_path}")
            calib_data = None

    if calib_data is None:
        print("No valid calibration data found. Starting calibration process...")
        calib_data = calibrate_stereo_cameras()
        if calib_data is None:
            print("Calibration failed or not enough images captured. Exiting.")
            exit()
        else:
            # Save the new calibration data
            np.savez(calib_file_path,
                     mtx_left=calib_data['mtx_left'], dist_left=calib_data['dist_left'],
                     mtx_right=calib_data['mtx_right'], dist_right=calib_data['dist_right'],
                     R=calib_data['R'], T=calib_data['T'], E=calib_data['E'], F=calib_data['F'])
            print(f"New calibration data saved to {calib_file_path}")

    # Initialize cameras for depth estimation
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    if not cap_left.isOpened():
        print("Error: Could not open left camera (index 0). Please check if the camera is connected and not in use. Exiting.")
        exit()
    if not cap_right.isOpened():
        print("Error: Could not open right camera (index 1). Please check if the camera is connected and not in use. Exiting.")
        exit()

    print("Cameras opened successfully for depth estimation.")
    print("Press 'q' to quit.")

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Error: Could not read frames. Exiting.")
            break

        avg_depth = find_chessboard_depth(frame_left, frame_right, calib_data)

        if avg_depth is not None:
            print(f"Average Chessboard Depth: {avg_depth:.2f} cm")
        else:
            print("Chessboard not detected or depth could not be calculated.")

        cv2.imshow('Left Camera Feed', frame_left)
        cv2.imshow('Right Camera Feed', frame_right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()