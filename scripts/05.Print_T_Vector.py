import numpy as np

# Load the .npz file
try:
    stereo_calib = np.load('stereo_calib.npz')
    
    # Access and print the T (Translation) vector
    if 'T' in stereo_calib:
        T = stereo_calib['T']
        print("Translation Vector (T):")
        print(T)
    else:
        print("Error: 'T' key not found in stereo_calib.npz")

except FileNotFoundError:
    print("Error: stereo_calib.npz file not found.")