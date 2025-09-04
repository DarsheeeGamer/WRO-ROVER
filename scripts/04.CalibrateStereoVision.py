import cv2
import glob
import numpy as np
import argparse

# --- Step 2a: Define Chessboard and Load Intrinsic Parameters ---
# Define the size of the chessboard corners
chessboardSize = (7,10) # Number of internal corners, e.g., for a 8 rowsx11 columns chess board

# Load the intrinsic calibration files for each camera
# Make sure these files exist from your single-camera calibration
parser = argparse.ArgumentParser(description='Perform stereo camera calibration.')
parser.add_argument('--left_calib_file', type=str, required=True, help='Path to the left camera calibration file (e.g., calib_left.npz).')
parser.add_argument('--right_calib_file', type=str, required=True, help='Path to the right camera calibration file (e.g., calib_right.npz).')
parser.add_argument('--left_image_dir', type=str, required=True, help='Directory containing left stereo images (e.g., stereo_images/left).')
parser.add_argument('--right_image_dir', type=str, required=True, help='Directory containing right stereo images (e.g., stereo_images/right).')
parser.add_argument('--output_file', type=str, required=True, help='Output file name for stereo calibration results (e.g., stereo_cal.npz).')
args = parser.parse_args()

left_calib = np.load(args.left_calib_file)
right_calib = np.load(args.right_calib_file)

cameraMatrixL = left_calib['cameraMatrix']
distL = left_calib['dist']
cameraMatrixR = right_calib['cameraMatrix']
distR = right_calib['dist']

# Define the object points (3D coordinates of the chessboard corners)
# Assuming a flat chessboard, Z=0.
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all stereo pairs
objpoints = []  # 3D points
imgpoints_left = []  # 2D points from left camera
imgpoints_right = [] # 2D points from right camera

# --- Step 2b: Find Corners in All Stereo Pairs ---
# Load all left and right images. Glob finds files based on a pattern.
left_images = sorted(glob.glob(f'{args.left_image_dir}/*.png'))
right_images = sorted(glob.glob(f'{args.right_image_dir}/*.png'))

# Ensure you have the same number of left and right images
if len(left_images) != len(right_images):
    print("Error: Number of left and right images do not match.")
    exit()

print(f"Found {len(left_images)} stereo image pairs.")

# A single image size is needed for stereo calibration.
img_size = None

for i in range(len(left_images)):
    # Read image pairs
    imgL = cv2.imread(left_images[i])
    imgR = cv2.imread(right_images[i])

    if imgL is None or imgR is None:
        print(f"Error: Could not read image pair {left_images[i]} and {right_images[i]}")
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = grayL.shape[::-1]

    # Find the chessboard corners in both images
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

    # If corners are found in *both* images, store the points
    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
        print(f"Corners found in pair {i+1}")
    else:
        print(f"Corners not found in one or both images of pair {i+1}")
        
print(f"\nSuccessfully found corners in {len(objpoints)} pairs.")

# --- Step 2c: Perform Stereo Calibration ---
# The function that finds the extrinsic parameters.
if len(objpoints) > 0:
    ret, cameraMatrixL, distL, cameraMatrixR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        cameraMatrixL,
        distL,
        cameraMatrixR,
        distR,
        img_size,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print("\nStereo Calibration successful!")
    print("\nRotation Matrix (R):")
    print(R)
    print("\nTranslation Vector (T):")
    print(T)
    print(f"\nReprojection Error: {ret}")

    # --- Step 2d: Save the Extrinsic Parameters ---
    np.savez(args.output_file, R=R, T=T, E=E, F=F)
    print(f"Extrinsic parameters saved to {args.output_file}")
else:
    print("Stereo calibration failed because no pairs of corners were found.")