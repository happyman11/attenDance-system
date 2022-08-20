import os
    
import openpyxl
import csv

from datetime import date
from datetime import datetime

path_attnd="C:/Users/Applications/Desktop/aTTANDANCE SYSTEM/CODE/Face-Recognizer-Using-LBPH-Recognizer-master/Face-Recognizer-Using-LBPH-Recognizer-master/atandance/"

files=os.listdir(path_attnd)
date_object = date.today() 
file_nme=str(date_object)+"attandance"+".csv"
Emp_id="Ravi"
Date=str(date_object)   
current_time = str(datetime.now().time())
full_path=path_attnd+file_nme 

if file_nme not in files:
   print(" Not Found")
   with open(full_path, 'w', newline='') as file:
       writer = csv.writer(file)
       
       writer.writerow(['Emp_id', 'Date', 'Time'])
       writer.writerow([Emp_id, Date, current_time])
                  
   
else:
   print("Found")
   with open(full_path, 'a', newline='') as file:
       writer = csv.writer(file)
       
      
       writer.writerow([Emp_id, Date, current_time])
   
