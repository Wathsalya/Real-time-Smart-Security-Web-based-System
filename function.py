import cv2
from flask import Flask, render_template, Response
# from datetime import datetime
from datetime import datetime, timedelta
import time
import requests
import numpy as np
# from matplotlib import pyplot as plt
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db
from flask_cors import CORS
import json

databaseURL = 'https://motiondetectpro-default-rtdb.asia-southeast1.firebasedatabase.app/'

cred_obj = firebase_admin.credentials.Certificate("motiondetectpro-firebase-adminsdk-xx63a-5842af1fd4.json")
default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL': databaseURL,
    'storageBucket': 'motiondetectpro.appspot.com'
})

bucket = storage.bucket()


def sendSms(count, imgUrl):
    #-------SMS---------
    # response = requests.get("https://app.smsapi.lk/sms/api?action=send-sms&api_key=Qnh3dGFyeUlwQk9Od0FESmpzRnY=&to=94719499838&from=Ticketz&sms=SecurityBreachDirected%20http://192.168.1.100:5000/img_threat")
    print('send sms test msg', count)
    token="5003082960:AAFSzKHnPHM8B9B_Nh7CEvVrFgdVOidOo_0"

    payload = {
        "photo": imgUrl,
        "caption": "Security Breach Detected !"
    }
    to_url = "https://api.telegram.org/bot{}/sendPhoto?chat_id=-768974533".format(token)
    resp = requests.post(to_url, data=payload)
    print("send telegram", resp)



def upload_img_to_firebase(frame):
    cv2.imwrite("ref.jpg", frame)
    img_url = 'ref.jpg'
    #img_url1 = ('%f.jpg' % timestamp)
    #img_url2 = ('images/%f.jpg' % timestamp)
    print("upload func")

    # upload to firebase ------------------------------

    blob = bucket.blob(img_url)
    outfile = img_url
    blob.upload_from_filename(outfile)

    bucket1 = storage.bucket(app=default_app)
    blob = bucket1.blob(img_url)

    return blob.generate_signed_url(timedelta(seconds=1200), method='GET')

    # ref = db.reference("/")
    # ref.push().set({
    # "Type": classNames[classId - 1],
    # "timestamp": timestamp,
    # "image": img_url1,
    # "run": "now"

    # })
