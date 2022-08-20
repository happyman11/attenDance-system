
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




with st.container():
 #navbar 
#https://bootsnipp.com/snippets/nNX3a     https://www.mockplus.com/blog/post/bootstrap-navbar-template
   components.html(
       """
       <link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
<script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<!------ Include the above in your HEAD tag ---------->
<nav class="navbar navbar-icon-top navbar-expand-lg navbar-dark bg-dark" >
  <a class="navbar-brand" href="https://www.rstiwari.com" target="_blank">Profile</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarSupportedContent">
    <ul class="navbar-nav mr-auto">
      <li class="nav-item active">
        <a class="nav-link" href=" https://tiwari11-rst.medium.com/" target="_blank">
          <i class="fa fa-home"></i>
          Medium
          <span class="sr-only">(current)</span>
          </a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href=" https://happyman11.github.io/" target="_blank">
          <i class="fa fa-envelope-o">
            <span class="badge badge-danger">Git Pages</span>
          </i>
         
        </a>
      </li>
      
        <li class="nav-item">
        <a class="nav-link" href="https://happyman11.github.io/" target="_blank">
          <i class="fa fa-globe">
            <span class="badge badge-success">Badges</span>
          </i>
         
        </a>
      </li>
          
        </a>
      </li>
      
      <li class="nav-item">
        <a class="nav-link disabled" href="https://ravishekhartiwari.blogspot.com/" target="_blank">
          <i class="fa fa-envelope-o">
            <span class="badge badge-warning">Blogspot</span>
          </i>
          
        </a>
      </li>
      
      
    </ul>
  
    
  </div>
</nav>
       """, height=70,
    )



cascadePath = "haarcascade_frontalface_default.xml"




def recogniser_funct(frame):
    
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('model/trained_model2.yml')

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    ID=[]
    Confd=[]
    
    for(x,y,w,h) in faces:
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 260, 0), 7)
        cv2.putText(frame, str(ID), (x,y-40),font, 2, (255,255,255), 3)
        
        ID.append(Id)
        Confd.append(conf)
        
        
    return (frame,Id,conf)
 

with st.container():

    col1, col2,col3 = st.columns(3)
    
    
    
    img_file_buffer = st.camera_input("Take Pictures")
    
    if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img_color = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            #cv2_img_grey = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)

        frame,ID,conf=recogniser_funct(cv2_img_color)
        
        with col1:
            
            st.image(frame, caption='Sunrise by the mountains')        
       
        



 
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