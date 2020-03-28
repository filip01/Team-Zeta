import numpy as np
import cv2
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: videoface video_file")
        sys.exit(1)

    min_size = (30,30)
    max_size = (60,60) # todo: determine face size
    haar_scale = 1.3
    min_neighbors = 5
    haar_flags = 0

    face_cascade = cv2.CascadeClassifier('/opt/ros/kinetic/share/OpenCV-3.3.1-dev/haarcascades/haarcascade_frontalface_default.xml')
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    num = 0
    tp = 0 # todo: count true positives... Assume that if 1 face is detected it is true positive?
    fp = 0
    fn = 0
    image_scale = None
    while(cap.isOpened()):
        num = num + 1
        ret, frame = cap.read()
        if not ret: break # Check if end of video

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 0)
        if image_scale == None:
            image_scale = frame.shape[1] / 240

        smallImage = cv2.resize(gray, (int(frame.shape[1] / image_scale), int(frame.shape[0] / image_scale)), interpolation=cv2.INTER_LINEAR)
        smallImage = cv2.equalizeHist(smallImage)

        # Detecting faces
        # todo: add min and max_size parameters
        faces = face_cascade.detectMultiScale(smallImage, haar_scale, min_neighbors, haar_flags)
        for (x,y,w,h) in faces:
            pt1 = (int(x * image_scale), int(y * image_scale))
            pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))

            frame = cv2.rectangle(frame,pt1,pt2,(255,0,0),2)

        num_detect = len(faces)
        if num_detect > 1: # Only one face per image
            fp = fp + num_detect - 1
        elif num_detect == 0: # Assume that every frame contains face
            fn = fn + 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print "number of frames: ", num
    print "false positives: ", fp
    print "false negatives: ", fn
    cap.release()
    cv2.destroyAllWindows()
