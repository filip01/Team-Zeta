import numpy as np
import cv2
import sys

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: videoface video_file init_distance final_distance")
        sys.exit(1)

    min_size = (30,30)
    max_size = (60,60) # todo: determine face size
    haar_scale = 1.3
    min_neighbors = 5
    haar_flags = 0

    face_cascade = cv2.CascadeClassifier('/opt/ros/kinetic/share/OpenCV-3.3.1-dev/haarcascades/haarcascade_frontalface_default.xml')
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    initial_distance = float(sys.argv[2])
    final_distance = float(sys.argv[3])
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_t = num_frames / float(fps)
    velocity = (final_distance - initial_distance) / video_t
    dist = 0.20
    # a list of: distance from, distance to, tp, fp, fn
    distance_data_list = [[0,0,0,0,0]] * int((initial_distance - final_distance) / dist)

    num = 0
    tp = 0 
    fp = 0
    fn = 0
    image_scale = None
    x = initial_distance
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret: break # Check if end of video

        num = num + 1
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
        # new data to be added to distance_data_list
        new_fp = 0
        new_fn = 0
        new_tp = 0
        if num_detect > 1: # Only one face per image
            fp = fp + 1
            new_fp = 1
        elif num_detect == 0: # Assume that every frame contains face
            fn = fn + 1
            new_fn = 1
        elif num_detect == 1: # Assume that if one face is detected it is true positive?
            tp = tp + 1
            new_tp = 1

        cv2.imshow('frame', frame)

        # Calculate current distance x
        t = num / float(fps)
        x = initial_distance + velocity * t

        distance_index = int((initial_distance - x) / dist)
        if distance_index == len(distance_data_list):
            distance_index -= 1

        [_,_,a,b,c] = distance_data_list[distance_index]
        interval = initial_distance - dist * distance_index
        distance_data_list[distance_index] = [interval, interval - dist, a + new_tp, b + new_fp, c + new_fn]

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    print "number of frames: ", num
    print "true positives: " , tp
    print "false positives: ", fp
    print "false negatives: ", fn
    print "table:"
    cap.release()
    cv2.destroyAllWindows()

    print "from,to,tp,fp,fn"
    for i in distance_data_list:
        print ','.join(map(str, i))
