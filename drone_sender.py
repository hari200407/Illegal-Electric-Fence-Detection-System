import cv2
import time
import requests
import json

SERVER_URL = "http://127.0.0.1:8080/frames"
DRONE_ID = "DRONE_01"

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Camera not opened"

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    telemetry = {
        "drone_id": DRONE_ID,
        "timestamp": time.time(),
        "lat": 12.3456,
        "lon": 78.9123,
        "alt": 20.0
    }

    _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    files = {'frame': ('frame.jpg', jpg.tobytes(), 'image/jpeg')}
    data = {'meta': json.dumps(telemetry)}

    try:
        resp = requests.post(SERVER_URL, files=files, data=data, timeout=5)
        result = resp.json()
        print(result)
    except Exception as e:
        print("Upload error:", e)

    time.sleep(1)
