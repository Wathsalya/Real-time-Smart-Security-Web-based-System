import cv2
from flask import Flask, render_template, Response, redirect
from datetime import datetime, timedelta
import time
from flask_cors import CORS
import function

app = Flask(__name__)
CORS(app)

info = {"person": "Not detect",
        "ses": "",
        "det_img_url": "",
        "timeCount": "",
        "conf": "",
        "move": ""}
# day time limit------
lim_1 = '06:00:00'  # 6 am
lim_2 = '18:00:00'  # 6 pm

msg_delay = 30  # delay(sec) for next msg


def gen():
    global info, lim_1, lim_2, msg_delay

    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture("./cctvde.mp4")#load a video stream

    person = 0
    sendMsg = 0

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))# reduce illumination changes in frames
    net.setInputSwapRB(True)#intialize a flag

    tic = 0
    timeRef = 0

    while True:
        # ret, image = video_capture.read()
        info['move'] = ""

        ret, frame1 = video_capture.read()
        ret, frame2 = video_capture.read()

        # success, img = cap.read()

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        FMT = '%H:%M:%S'
        tdelta = datetime.strptime(current_time, FMT)
        t_lim_1 = datetime.strptime(lim_1, FMT)
        t_lim_2 = datetime.strptime(lim_2, FMT)

        if t_lim_1 < tdelta < t_lim_2:

            cv2.imwrite("fr.jpg", frame1)
            # ------------------------------------------day mode----------------------------------------
            # print("Current Time = day")

            info['ses'] = "Day"

            difference = cv2.absdiff(frame1, frame2)
            difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)# to apply gaussian blur
            gussBlur = cv2.GaussianBlur(difference_gray, (5, 5), 0) #Noise removing
            _, thresh = cv2.threshold(gussBlur, 20, 255, cv2.THRESH_BINARY) #Smoothing and convert to black and white
            dilated = cv2.dilate(thresh, None, iterations=3) # reduce the noisy inside the edge region
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# coner values

            classIds, confs, bbox = net.detect(frame1, confThreshold=0.45)

            for contour in contours:
                info['move'] = "Movement Detect!"
                (x, y, w, h) = cv2.boundingRect(contour)
                # print(contour, "next")
                if cv2.contourArea(contour) < 900:
                    continue
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if len(classIds) != 0:
                    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):

                        info['conf'] = round(confidence * 100, 2)
                        if classNames[classId - 1] == 'person':
                            info['person'] = "detect" #json key person det to detect

                            cv2.rectangle(frame1, box, color=(0, 0, 255), thickness=2)
                            cv2.putText(frame1, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 20),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                            cv2.imwrite("fr.jpg", frame1)

                            person += 1

            if person > 5 and sendMsg == 0:
                resUrl = function.upload_img_to_firebase(frame1)
                info['det_img_url'] = resUrl
                sendMsg = 1
                function.sendSms(person, resUrl)
                # SMS API---------------------------

            tic = time.perf_counter()
            info['timeCount'] = round(tic, 2) - round(timeRef, 2)
            if tic - timeRef > msg_delay:  # seconds - delay for next msg and upload
                sendMsg = 0
                person = 0
                timeRef = tic

                info['person'] = "No Security Threat"

            if cv2.waitKey(50) == 27:
                break
        else:
            # ------------------------------------------night mode----------------------------------------
            # print("Current Time = night")
            info['ses'] = "Night"

            ret, t1p = video_capture.read()
            ret, tp = video_capture.read()

            t1 = cv2.cvtColor(t1p, cv2.COLOR_RGB2GRAY)
            t = cv2.cvtColor(tp, cv2.COLOR_RGB2GRAY)

            d1 = cv2.absdiff(t, t1)

            m_result = cv2.equalizeHist(d1)
            # m_result = diffImg

            # output = cv2.normalize(m_result, None, 0, 255, cv2.NORM_MINMAX)

            blur = cv2.GaussianBlur(m_result, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            classIds, confs, bbox = net.detect(t1p, confThreshold=0.45)

            for contour in contours:
                info['move'] = "Movement Detect!"
                # (x, y, w, h) = cv2.boundingRect(contour)
                # print(contour, "next")
                if cv2.contourArea(contour) < 900:
                    continue
                # cv2.rectangle(f1p, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if len(classIds) != 0:
                    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                        info['conf'] = round(confidence * 100, 2)

                        if classNames[classId - 1] == 'person':
                            info['person'] = "detect"

                            cv2.rectangle(t1p, box, color=(0, 100, 255), thickness=2)
                            cv2.putText(t1p, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 20),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                            cv2.imwrite("fr.jpg", t1p)

                            person += 1

            if person > 4 and sendMsg == 0:
                resUrl = function.upload_img_to_firebase(frame1)
                info['det_img_url'] = resUrl

                sendMsg = 1

                # function.sendSms(person, resUrl)

            tic = time.perf_counter()
            info['timeCount'] = round(tic, 2) - round(timeRef, 2)
            if tic - timeRef > msg_delay:  # 20 sec - delay for next msg and upload
                sendMsg = 0
                person = 0
                timeRef = tic
                info['person'] = "No Security Threat"

            if cv2.waitKey(50) == 27:
                break
            frame1 = t1p

        cv2.imwrite('fr.jpg', frame1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('fr.jpg', 'rb').read() + b'\r\n')
    video_capture.release()


@app.route('/')
def index():
    """Video streaming"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/info', methods=['GET'])
def data():
    return info


@app.route("/img_threat")
def img_url():
    url = info['det_img_url']
    return redirect(url, code=302)


if __name__ == '__main__':
    app.run()
    # app.run(host='192.168.1.100', port='5000', debug=False)
