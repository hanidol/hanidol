import cv2
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

id=input('enter the user id');
sampleNum=0;
while(cv2.waitKey(1)!=27):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = plate_cascade.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataset/User."+str(id)+"."+str(sampleNum)+".png",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,225),2)
        cv2.waitKey(100);
    cv2.imshow("Face",img);
    cv2.waitKey(1);
    if(sampleNum>100):
        break;
cam.release()
cv2.destroyAllWindows()
