import cv2
import csv
import os
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
#pip install opencv-contrib-python

#####Window is our Main frame of system
st.set_page_config(layout="wide")








#pip install opencv-contrib-python

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/trained_model2.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

def recogniser_funct():

    cam = cv2.VideoCapture(0)
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        #print(len(faces))
        if ((len(faces)) == 1):
            for(x,y,w,h) in faces:
                Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 1)
                cv2.putText(im, str(Id), (x,y-40),font, 2, (255,255,255), 3)
		
            cv2.imshow('im',im)
            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
            
        else:
            msg="More than 1 Face detected"
            print(msg)
            cv2.putText(im, str(msg), (50,50),font, 2, (255,110,110), 10)
            cv2.imshow('im',im)
            
    cam.release()
    cv2.destroyAllWindows()
    


def take_mannually():

    with st.container():
        col1,col2 =st.columns(2) 

        with col1:
            img_file_buffer = st.camera_input("Take a picture")
   
            if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
                gray=cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
                faces=faceCascade.detectMultiScale(gray, 1.2,5)
        #print(len(faces))
                if ((len(faces)) == 1):
                    for(x,y,w,h) in faces:
                        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 1)
                        cv2.putText(im, str(Id), (x,y-40),font, 2, (255,255,255), 3)
		        
                    with col2:
                        st.image(im,alt="Candidates Image")                    
                  
            
                else:
                    msg="More than 1 Face detected"
                    print(msg)
                    cv2.putText(im, str(msg), (50,50),font, 2, (255,110,110), 10)
         

with st.container(): 

 col0,col1, col2 = st.columns([1,3,3])
 
 with col1:
 
    st.button("Proceed to take attendance automatically", on_click=recogniser_funct)
      
 with col2:
 
    st.button("Attandance by taking Images",on_click=take_mannually)

 
   
 


        



 
with st.container():
    components.html(
     """
     <div style="position: fixed;
   left: 0;
   bottom: 0;
   width: 100%;
   background-color: black;
   color: white;
   text-align: center;">
  <p>Ravi Shekhar Tiwari</p>
</div>
     """,height=140,)