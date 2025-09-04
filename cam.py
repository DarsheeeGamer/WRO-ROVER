import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

# Use only cam 0 and cam 2
cam_indices = [0, 2]
caps = [cv2.VideoCapture(idx, cv2.CAP_V4L2) for idx in cam_indices]

def gen_frames():
    while True:
        frames = []
        for cap in caps:
            if cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frames.append(frame)

        # Resize all frames to same height
        target_height = min(f.shape[0] for f in frames)
        resized = [cv2.resize(f, (int(f.shape[1]*target_height/f.shape[0]), target_height)) for f in frames]

        # Combine horizontally
        combined = np.hstack(resized)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', combined)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/")
def index():
    return """
    <html>
        <head><title>Dual Cam Stream</title></head>
        <body>
            <h1>Dual Camera Stream (0 + 2)</h1>
            <img src="/video_feed" width="100%">
        </body>
    </html>
    """

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # Start the FastAPI server automatically
    uvicorn.run(app, host="0.0.0.0", port=6969)
