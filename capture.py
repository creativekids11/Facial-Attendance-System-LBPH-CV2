import cv2
import os

cap=cv2.VideoCapture(0)
faceClassifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img_id=0
img_index=0
name=input("client name: ")
for ids,_ in enumerate(os.listdir("imgs/")):
    if ids>99:
        ids=int(ids)/100+1
        img_index=ids

if img_index==0:
    img_index+=1

while True:
    _,img=cap.read()
    key=cv2.waitKey(1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces=faceClassifier.detectMultiScale(gray, 1.9, 5)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_cropped=img[y:y+h,x:x+w]
        img_id+=1 
        print(str(img_index))
        print(img_id)
        face=cv2.resize(face_cropped,(450,450))
        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        path="imgs/"+name+"."+str(int(img_index))+"."+str(int(img_id))+".jpg"
        cv2.imwrite(path,face)
        cv2.putText(img, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 
                   2, (0,255,0), 2, cv2.LINE_AA)

    # Display the output
    cv2.imshow('img', img)


    if key==113 or img_id==100:
        break
