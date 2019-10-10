import cv2
import numpy as np
import matplotlib.pyplot as plt

cap=cv2.VideoCapture(0)

#cv2.waitKey(0)
#cap.release()

faceCascade=cv2.CascadeClassifier('haarcascade.xml')

faceData=[]
faceCount=0

ret,frame=cap.read()
cap.release()
grayFace=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

plt.imshow(grayFace,cmap='gray')

faces=faceCascade.detectMultiScale(grayFace,1.3,5)

names={
       1:"deekshitha",
       0:"hencel"
       }
#x,y,w,h=faces[0,:]
i=0
for (x,y,w,h) in faces:
    output=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    name=names[i]
    cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
    i+=1
plt.imshow(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))

'''
f=faces[0]
a=frame[f[1]:(f[1]+f[3]),f[0]:(f[0]+f[2]),:]
plt.imshow(a,cmap='gray')
'''





