import cv2 as cv
import time


video = cv.VideoCapture(0) #start recording --here argument is the camera index ("0" for built-in cam).

start_time = time.time()

while True:

    ret, frame = video.read()


    cv.imshow("video", frame)



    if time.time() - start_time > 2.5: #logic for recording for particular amount of time. (2.5 secs in this case)
        break

    if cv.waitKey(1) & 0xFF == ord("q"): #press "q" for force quit. 
        break
    

if __name__ == "__main__":
    
    print("Capturing video")

