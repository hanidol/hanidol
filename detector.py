#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import requests
import time

#plate_cascade =cv2.CascadeClassifier('DATA/haarcascades/india_license_plate.xml')# Loads the data required for detecting the license plates from cascade classifier.
plate_cascade =cv2.CascadeClassifier('license_plate.xml')
def detect_plate(img): # the function detects and perfors blurring on the number plate.
    plate_img = img.copy()
    roi = img.copy()
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7) # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    
    for (x,y,w,h) in plate_rect:
        
        roi_ = roi[y:y+h, x:x+w, :] # extracting the Region of Interest of license plate for blurring.
        blurred_roi = cv2.blur(roi_, ksize=(16,16)) # performing blur operation on the ROI
        plate_img[y:y+h, x:x+w, :] = blurred_roi # replacing the original license plate with the blurred one.

        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3) # finally representing the detected contours by drawing rectangles around the edges.
        
    return plate_img # returning the processed image.




cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("parck/trainingData.yml");
id = 0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 1
fontcolor = (0,511,1)
count=0
c=0
while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = plate_cascade.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,180),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        print(conf);
        if(id==1,2,3,4,5 and conf<50):
               id="You Welcome"       
        else:
            id="Stranger"
            count=count+1
        if(count>30):
            if(c!=1):
                count=0
                c=1
                print("Alarm On")
                time.sleep(10)     
        cv2.putText(img,str(id),(x,y+h+25),fontface,fontsize,fontcolor,2);
        print(id)
    cv2.imshow("Face",img);

    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
