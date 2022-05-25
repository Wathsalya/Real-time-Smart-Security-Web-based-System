# camera.py
# import the necessary packages
import cv2
import cv2 as cv
from datetime import datetime
import time

import numpy as np
#from matplotlib import pyplot as plt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db

# defining face detector
face_cascade = cv2.CascadeClassifier("a.xml")
ds_factor = 0.6


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()


    def get_frame(self):
        # extracting frames
        # ret, frame = self.video.read()
        #frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
          #                 interpolation=cv2.INTER_AREA)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        #for (x, y, w, h) in face_rects:
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
         #   break
        # encode OpenCV raw frame to jpg and displaying it

        # img = cv2.imread("d2.jpg")

        #ret, jpeg = cv2.imencode('.jpg', frame)

        # ret, jpeg2 = cv2.imencode('.jpg', img)

        #################################################################

        thres = 0.45  # Threshold to detect object

        # cap = cv.VideoCapture("./CCTV1.mp4")
        cap = cv.VideoCapture(0)
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        classNames = []
        person = 0
        sendMsg = 0
        classFile = 'coco.names'
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'

        net = cv.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        tic = 0
        timeRef = 0
        while cap.isOpened():

            # success, img = cap.read()
            classIds, confs, bbox = net.detect(frame1, confThreshold=thres)

            diff = cv.absdiff(frame1, frame2)
            diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
            _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
            dilated = cv.dilate(thresh, None, iterations=3)
            contours, _ = cv.findContours(
                dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x, y, w, h) = cv.boundingRect(contour)
                # print(contour, "next")
                if cv.contourArea(contour) < 900:
                    continue
                cv.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv.FONT_HERSHEY_SIMPLEX,
                           1, (255, 0, 0), 3)

                if len(classIds) != 0:
                    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

                        # cv.putText(frame1, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        # cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                        if classNames[classId - 1] == 'person':
                            now = datetime.now()
                            timestamp = datetime.timestamp(now)

                            cv.rectangle(frame1, box, color=(0, 0, 255), thickness=2)
                            cv.putText(frame1, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 20),
                                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                            # cv.imwrite("%f.jpg" % timestamp, frame1)
                            cv.imwrite("fr.jpg", frame1)
                            # cv2.imshow('frame', frame)
                            # print("human")
                            print(classNames[classId - 1])
                            person += 1

            ret, jpeg = cv2.imencode('.jpg', frame1)

            return jpeg.tobytes()

            # cv.drawContours(frame1, contours, -1, (0, 255, 0), 2)

            #cv.imshow("Video", frame1)
            # cv.imshow("Video2", diff_gray)
            frame1 = frame2
            ret, frame2 = cap.read()

            if person > 5 and sendMsg == 0:
                sendMsg = 1
                print("send msg", person)
                # cv.imwrite("%f.jpg" % timestamp, frame1)
                cv.imwrite("ref.jpg", frame1)
                img_url = 'ref.jpg'
                img_url1 = ('%f.jpg' % timestamp)
                img_url2 = ('images/%f.jpg' % timestamp)

                print("upload")

                # blob = bucket.blob(img_url2)
                # outfile = img_url
                # blob.upload_from_filename(outfile)

                #ref = db.reference("/")
               # ref.push().set({
                    # "Type": classNames[classId - 1],
                   # "timestamp": timestamp,
                   # "image": img_url1

               # })
            tic = time.perf_counter()
            print(f" {tic - timeRef:0.4f}")
            if tic - timeRef > 30:  # 10 min
                sendMsg = 0
                person = 0
                timeRef = tic

            if cv.waitKey(50) == 27:
                break

