#!/usr/bin/env python3
"""
Capture stereo PAIRS from two USB cameras.

Usage:
  python3 03.CaptureStereoImages.py --left 0 --right 1 \
    --out_left left_stereo --out_right right_stereo \
    --width 1280 --height 720

Controls:
- SPACE → Save current stereo pair
- ESC   → Quit
"""

import argparse, os, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=int, default=0, help="Left cam index")
    ap.add_argument("--right", type=int, default=1, help="Right cam index")
    ap.add_argument("--out_left", type=str, default="left_stereo")
    ap.add_argument("--out_right", type=str, default="right_stereo")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    os.makedirs(args.out_left, exist_ok=True)
    os.makedirs(args.out_right, exist_ok=True)

    capL = cv2.VideoCapture(args.left)
    capR = cv2.VideoCapture(args.right)
    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Press SPACE to save stereo pair, ESC to quit.")
    idx = 0

    while True:
        retL, imgL = capL.read()
        retR, imgR = capR.read()
        if not (retL and retR):
            print("[ERR] Camera read failed")
            break

        vis = cv2.hconcat([imgL, imgR])
        cv2.putText(vis, "SPACE: save pair  |  ESC: quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Stereo Capture (L | R)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == 32: # SPACE
            lp = os.path.join(args.out_left,  f"left_{idx:04d}.png")
            rp = os.path.join(args.out_right, f"right_{idx:04d}.png")
            cv2.imwrite(lp, imgL)
            cv2.imwrite(rp, imgR)
            print("[SAVE]", lp, " | ", rp)
            idx += 1

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
