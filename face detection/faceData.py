import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

cap=cv2.VideoCapture(0)


faceCascade=cv2.CascadeClassifier('haarcascade.xml')

faceData=[]
faceCount=0

while True:
    ret,frame=cap.read()
    if ret == True:
        grayFace=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(grayFace,1.3,5)
        
        for (x,y,w,h) in faces:
            croppedFace=frame[y:y+h,x:x+w,:]
            resizedFace=cv2.resize(croppedFace,(50,50))
            faceData.append(resizedFace)
            #cv2.imwrite('sample'+str(faceCount)+'.jpg',resizedFace)
            faceCount+=1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('capturing Frames',frame)
            #cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
            #plt.imshow(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
        #    time.sleep(0)
            
        
        if cv2.waitKey(1)==27 or len(faceData)>=10:
            break
        else:
            print("camera error")
            
cap.release()
cv2.destroyAllWindows()

faceData=np.asarray(faceData)

np.save('Face matched',faceData)

'''
with open("output.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(faceData)
'''

