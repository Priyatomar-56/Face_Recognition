import cv2

import numpy as np

import face_recognition  
import os  # to import images we are using os library 
from datetime import datetime
 
path = 'image-attendance'# the folder of the images
images = [] 
classNames = [] 
myList = os.listdir(path) # to get the images from the list  
print(myList) 
 
for cl in myList: # cl is the name of our image,for example first image is virat
    curImg = cv2.imread(f'{path}/{cl}') # to read image from the path
    images.append(curImg) 
    classNames.append(os.path.splitext(cl)[0]) #to get the name without the jpg we use splitext.
print(classNames)
 
def findEncodings(images): # defining the function
    encodeList = [] 
    for img in images: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        encode = face_recognition.face_encodings(img)[0] # to find the encoding
        encodeList.append(encode) 
    return encodeList  
     
def markAttendance(name):  
    with open('attendance.csv', 'r+') as f:# open attendance.csv file 
        mydataList = f.readlines() 
        print(mydataList)  
        nameList = [] 
        for line in mydataList: 
            entry =  line.split(',')  
            nameList.append(entry[0])
        if name not in nameList: 
            now = datetime.now() 
            #print("now")
            dtString = now.strftime('%H:%M:%S') 
            #print("the time is",dtString) 
            f.writelines(f'\n{name},{dtString}') 

         
#markAttendance('Virat kohli')

     
encodeListKnown = findEncodings(images) # calling of the function
#print(len(encodeListKnown)) 
 
print('Encoding complete') 
 
cap = cv2.VideoCapture(0) # to get the image to comapre from the webcam (computer camera)
 
while True: 
    success, img = cap.read() # reading the image from the webcam inout
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25) # to reduce the size og the image
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)# to convert the image into rgb
  
    facesCurFrame = face_recognition.face_locations(imgSmall)# to find the loaction of the faces
    encodeCurFrame = face_recognition.face_encodings(imgSmall,facesCurFrame) 
     
    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame): 
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) 
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) 
        #print(faceDis) 
        matchIndex = np.argmin(faceDis)# to fetch the minimum face dis image 

        if matches[matchIndex]: 
            name = classNames[matchIndex].upper() 
            print(name)  
            y1,x2,y2,x1 = faceLoc 
            y1 ,x2 ,y2 ,x1 = y1*4 , x2*4 , y2*4 , x1*4 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2) 
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) 
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2) 
            markAttendance(name)
             
    cv2.imshow('Webcam',img) 
    cv2.waitKey(1)
    
    #C:\Users\HP\OneDrive\Desktop\face-recognition\image-attendance
