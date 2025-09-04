#!/usr/bin/env python3
"""
Capture calibration images from a single USB camera.

Usage:
  python3 01.CaptureCalibrationImages.py --cam 0 --out left_calib --width 1280 --height 720

Controls:
- SPACE → Save current frame as an image
- ESC   → Quit
"""

import argparse, os, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Video device index")
    ap.add_argument("--out", type=str, default="left_calib", help="Output folder")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("[INFO] Press SPACE to save a photo, ESC to quit.")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Camera read failed")
            break

        cv2.putText(frame, "SPACE: save  |  ESC: quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Calibration Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            path = os.path.join(args.out, f"img_{idx:04d}.png")
            cv2.imwrite(path, frame)
            print("[SAVE]", path)
            idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
