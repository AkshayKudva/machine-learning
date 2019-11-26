import cv2 
import numpy as np

faceCascade=cv2.CascadeClassifier('haarcascade.xml')
person1=np.load('deek.npy').reshape(20,50*50*3)
person2=np.load('deepa.npy').reshape(20,50*50*3)

names={
      0:'face1',
      1:'face2'
      }
data=np.concatenate([person1,person2])
labels=np.zeros((40,1))
labels[:21,:]=0.0
labels[21:,:]=1.0



def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(testInput,data,labels,k):
    numRows=data.shape[0]
    dist=[]
    for i in range(numRows):
        dist.append(distance(testInput,data[i]))
    dist=np.asarray(dist)
    indx=np.argsort(dist)
    sortedLabels=labels[indx][:k]
    counts=np.unique(sortedLabels,return_counts=True)
    return counts[0][np.argmax(counts[-1])]

sample=[7,1,8,2,9,3]
sample=np.array(sample).reshape(3,2)
sampleL=[1,1,0]
sampleL=np.array(sampleL).reshape(3,1)
sampleI=[3,4]
sampleI=np.array(sampleI).reshape(1,2)

pred=knn(sampleI,sample,sampleL,3)

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    grayFace=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(grayFace,1.5,5)
    
    for (x,y,w,h) in faces:
        croppedFace=frame[y:y+h,x:x+w,:]
        resizedFace=cv2.resize(croppedFace,(50,50))
        prediction=knn(resizedFace.flatten(),data,labels,5)
        name=names[int(prediction)]
        cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow('Face Recognition',frame)
    
    if(cv2.waitKey(1)==27):
        break
cap.release()
cv2.destroyAllWindows()

    


