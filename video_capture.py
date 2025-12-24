import cv2 as cv
import mediapipe as mp 
import time


video = cv.VideoCapture(0)
start_time = time.time()

while True:

    ret, frame = video.read()


    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow("video", img)



    if time.time() - start_time > 2.5:
        break

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    

if __name__ == "__main__":
    
    print("Capturing video")

