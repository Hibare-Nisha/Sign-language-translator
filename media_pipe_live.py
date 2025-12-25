import mediapipe as mp
import cv2 as cv
import time
import csv
import os

# --- Setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = r"C:/Users/chann/major_project/hand_landmarker.task"

# Connection map for drawing lines between joints
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
]

# --- CSV Header Initialization ---
csv_file = "landmarks.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Create headers for multiple hands if needed, or keep standard 21
        header = ["hand_idx"] + [coord for i in range(21) for coord in (f"x{i}", f"y{i}", f"z{i}")]
        writer.writerow(header)

# Global result storage
latest_result = None

def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# --- KEY FIX: Added num_hands=2 ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,  # <--- This allows detection of both hands
    result_callback=result_callback
)

with HandLandmarker.create_from_options(options) as landmarker:
    vid = cv.VideoCapture(0)
    prev_time = time.time()

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        
        # Mirror the frame for a more natural "selfie" view
        frame = cv.flip(frame, 1)

        # MediaPipe needs RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Send to AI
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        h, w, _ = frame.shape

        if latest_result and latest_result.hand_landmarks:
            # Loop through EVERY hand detected (1 or 2)
            for idx, landmarks in enumerate(latest_result.hand_landmarks):

                # Draw Connections
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                for s, e in HAND_CONNECTIONS:
                    cv.line(frame, pts[s], pts[e], (255, 0, 0), 2)
                
                # Draw Landmarks
                for pt in pts:
                    cv.circle(frame, pt, 5, (0, 255, 0), -1)
        else:
            cv.putText(frame, "No Hands Detected", (50, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / max(1e-6, (curr_time - prev_time))
        prev_time = curr_time
        cv.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv.imshow("Hand Tracking", frame)

 #press q to exit.
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC key
            print("Closing application...")
            break

    vid.release()
    cv.destroyAllWindows()