import cv2

import numpy as np

import face_recognition 

imgVirat= face_recognition.load_image_file('virat.jpg')
imgVirat=cv2.cvtColor(imgVirat,cv2.COLOR_BGR2RGB)
 
imgTest = face_recognition.load_image_file('virat-kohli.webp') 
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB) 
 
faceLoc = face_recognition.face_locations(imgVirat)[0]  
encodeVirat = face_recognition.face_encodings(imgVirat)[0]  

cv2.rectangle(imgVirat,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)  

faceLocTest = face_recognition.face_locations(imgTest)[0]  
encodeTest = face_recognition.face_encodings(imgTest)[0]  

cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)  

#comparing image 
results = face_recognition.compare_faces([encodeVirat],encodeTest) 

#print(faceLoc) # it will print fours points top,bottom,left,right 
faceDis = face_recognition.face_distance([encodeVirat],encodeTest) 
print(results,faceDis) 
cv2.putText(imgVirat,f'{results} { round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.putText(imgTest,f'{results} { round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) 
cv2.imshow('virat kohli',imgVirat)  
cv2.imshow('virat test',imgTest) # imshow method is used to display an image in a window
cv2.waitKey(0) #it will display the window infinitely until any keypress.waitkey(1) will display a freame foe 1ms. 

 



   

   




         






  