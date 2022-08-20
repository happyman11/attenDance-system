
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
from numpy.random import randint

#Variables

Employee_name_check=1
Employee_number_check=1
Path_dataset="C:/Users/Applications/Desktop/aTTANDANCE SYSTEM/CODE/LBPH -Streamlit/dataset/"
Path_Harcascade="C:/Users/Applications/Desktop/aTTANDANCE SYSTEM/CODE/LBPH -Streamlit/haarcascade_frontalface_default.xml"
Path_model="C:/Users/Applications/Desktop/aTTANDANCE SYSTEM/CODE/LBPH -Streamlit/model/trained_model2.yml"

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
 #save images automatically
 
def take_img(Name_Employee,Number_Employee,samples):

   
    sampleNum = 0
    
    ID = str(Number_Employee)
    Name = str(Name_Employee)
    
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(Path_Harcascade)
    
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            Path_dataset_image =Path_dataset + Name + "." + ID + '.' + str(sampleNum) + ".jpg"                   
            cv2.imwrite(Path_dataset_image,gray[y:y + h, x:x + w])
            cv2.imshow('Frame', img)
                # wait for 100 miliseconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                # break if the sample number is morethan 100
        elif sampleNum > 200:
            break
    cam.release()
    cv2.destroyAllWindows()
        
    res = "Images Saved: ID" + ID + " Name : " + Name
  
    st.success(res)
    st.balloons()
      
 

## take pictures for dataset Path_dataset mannually


def save_dataset(Name_Employee,Number_Employee):
    ID = str(Number_Employee)
    Name = str(Name_Employee)
    sampleNum= randint(0, 2000, 1)
    sampleNum = sampleNum + 1
    Path_dataset_image =Path_dataset + Name + "." + ID + '.' + str(sampleNum) + ".jpg"
    
    img_file_buffer = st.camera_input("Create Dataset")
    
    if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        time.sleep(3)
        cv2_img_color = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv2_img_grey = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_GRAYSCALE)
         
        detector = cv2.CascadeClassifier(Path_Harcascade)
        faces = detector.detectMultiScale(cv2_img_grey, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(cv2_img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite(Path_dataset_image,cv2_img_grey[y:y + h, x:x + w])
        
        
        res = "Images Saved: ID" + ID + " Name : " + Name
        st.success(msg)
        st.balloons()

def getImagesAndLabels(path,detector):
    imagePaths = os.listdir(path)
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    Name=[]
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = cv2.imread(Path_dataset+imagePath, 0)
         
        # Now we are converting the PIL image into numpy array
       
        
        # getting the Id from the image
        
        faces = detector.detectMultiScale(pilImage, 1.3, 5)
        # If a face is there then append that in the list as well as Id of it
        

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        name= str( imagePath.split(".")[0])
        for (x, y, w, h) in faces:
            faceSamples.append(pilImage[y:y + h, x:x + w])
            Ids.append(Id)
            Name.append(name)
    return faceSamples, Ids,Name




def trainimg():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    try:
        global faces,Id, Name
        faces, Id,Name = getImagesAndLabels(Path_dataset,detector)
    except Exception as e:
        l='please make "dataset" folder & put Images'
        st.error(l)
        
    with st.spinner('Model is Trainning'):
               time.sleep(5)
    recognizer.train(faces, np.array(Id)) 
    try:
        recognizer.save(Path_model)
    except Exception as e:
        q='Please make "model" folder'
        st.error(q)

    res = "Model Trained"  # +",".join(str(f) for f in Id)
    st.success(res)
   
    st.balloons()


with st.container():

    col1, col2,col3 = st.columns(3)
    
    with col1:
        option1 = st.selectbox('How would you like to save dataset',
                               ('Automatically','Mannualy'))    
        
    with col2:
        Number_Employee = st.text_input('Employee Number')   
        
    with col3:
        
        Name_Employee = st.text_input('Employee Name')
   
    
        
       


    
    if (len(Name_Employee)>Employee_name_check and len(Number_Employee) >Employee_number_check):
                
                
                
        if (option1 == 'Automatically'):
            
            samples_saved = st.slider('Number of damples you want to save?', 0, 500, 20)
            time.sleep(5)
            if (samples_saved == "0"):
                with st.spinner('You have 5 sec to select the sample size else 20 samples will be generated automatically'):
                    time.sleep(5)
                    
                    st.success('Generating for 20 samples')
            
            st.write(samples_saved)
            
            with st.spinner(text="Get ready for dataset collection Automatically. Starts in 3 sec"):
                time.sleep(5)
                
            take_img(Name_Employee,Number_Employee,samples_saved)
            Train_Checkbox = st.checkbox('Dataset Uploaded?? Train your model')
            if Train_Checkbox:
                trainimg()
                         
        else:
        
            with st.spinner(text="Get ready for dataset collection Mannually. Starts in 3 sec"):
                time.sleep(5)
            save_dataset(Name_Employee,Number_Employee)
        
            Train_Checkbox = st.checkbox('Dataset Uploaded?? Train your model')
            if Train_Checkbox:
                trainimg()
           
    
    else:
        
        if ( len(Name_Employee)<Employee_name_check):
            msg="Please Check Employee Name"
            st.error(msg)
            
    
        elif (len(Number_Employee) <Employee_number_check):
            msg="Please Check Employee Number"
            st.error(msg)


#  #footer

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