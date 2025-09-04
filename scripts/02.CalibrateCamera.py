import cv2
import glob
import numpy as np
import argparse

# Step 2a: Define Chessboard and Object Points
# Define the size of the chessboard squares. These are the corners *between* the squares.
# For a 8x11 board, there are 7x10 interior corners. Adjust these numbers based on your pattern.
chessboardSize = (7,10)

# Define the size of one chessboard square in a real-world unit (e.g., cm).
# This is crucial for obtaining real-world camera parameters.
squareSize = 2.5 # In cm

# Create an array to store real-world 3D points of the chessboard corners.
# For a flat pattern, the z-coordinate is always 0.
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * squareSize

# Create empty lists to store the real-world (3D) and image (2D) points.
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Step 2b: Find Chessboard Corners in All Images
# Load all the captured images from the folder.

parser = argparse.ArgumentParser(description='Calibrate a single camera using chessboard images.')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing calibration images (e.g., calibration_images/left).')
parser.add_argument('--output_file', type=str, required=True, help='Output file name for calibration results (e.g., calib_left.npz).')
args = parser.parse_args()

images = glob.glob(f'{args.input_dir}/*.png') # Changed to .png as per previous capture

# Define a counter for visualization
count = 0

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

    # If corners are found, add object points and image points to the lists.
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Optional: Draw and display the corners to check the detection
        # You can uncomment this to verify the corner finding process.
        # cv2.drawChessboardCorners(img, chessboardSize, corners, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)
        # count += 1
        # print(f"Processing image {count}")
    else:
        print(f"Corners not found in {image}")

# Optional: Clean up windows
# cv2.destroyAllWindows()

# Step 2c: Calibrate the Camera
# This is the main function that performs the calibration.
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Step 2d: Print and Save the Results
print("Calibration successful!")
print("\nCamera Matrix (Intrinsic Parameters):")
print(cameraMatrix)
print("\nDistortion Coefficients:")
print(dist)

# Save the calibration results to a file for later use
np.savez(args.output_file,
         cameraMatrix=cameraMatrix,
         dist=dist,
         rvecs=rvecs,
         tvecs=tvecs)