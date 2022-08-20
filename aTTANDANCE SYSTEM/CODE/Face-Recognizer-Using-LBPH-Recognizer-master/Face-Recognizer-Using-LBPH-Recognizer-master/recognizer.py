import cv2
import numpy as np
import os
    
import openpyxl
import csv
import time
from datetime import date
from datetime import datetime
#pip install opencv-contrib-python

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/trained_model2.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
path_attnd="C:/Users/Applications/Desktop/aTTANDANCE SYSTEM/CODE/Face-Recognizer-Using-LBPH-Recognizer-master/Face-Recognizer-Using-LBPH-Recognizer-master/atandance/"

def lodge_attandance(Emp_id):

    files=os.listdir(path_attnd)
    date_object = date.today() 
    file_nme=str(date_object)+"attandance"+".csv"
    Emp_id=str(Emp_id)
    Date=str(date_object)   
    current_time = str(datetime.now().time())
    full_path=path_attnd+file_nme 
    
    if file_nme not in files:
        
        with open(full_path, 'w', newline='') as file:
        
             writer = csv.writer(file)
       
             writer.writerow(['Emp_id', 'Date', 'Time'])
             writer.writerow([Emp_id, Date, current_time])
    
    else:
        
        with open(full_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([Emp_id, Date, current_time])
    return (current_time)

cam = cv2.VideoCapture(0)
while True:
    
    ret, im =cam.read()
     
    if not ret:
        break    
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    
    for(x,y,w,h) in faces:
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        #stamp=lodge_attandance(Id)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 1)
        
        cv2.putText(im,str(Id), (x,y-40),font, 1, (255,255,255), 1)
        
    cv2.imshow('Attandance window',im)
    time.sleep(3)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()