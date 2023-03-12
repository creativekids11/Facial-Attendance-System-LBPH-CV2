import os
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('attSysClassifier.xml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0

# add the list of names of your dataset here
path=[ os.path.join("imgs/", file) for file in os.listdir("imgs/") ]
names=[]
for image in path:
    name=image.split(".")[0]
    name=name.split("/")[1]
    names.append(name)
   

names=list(set(names))
print(names)


cam = cv2.VideoCapture(0)

while True:
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.9,
        minNeighbors = 5
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(id)
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id-1]
            confidence = "  {0}%".format(round(100 - confidence))

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        

        if id!="Unknown":
            import pandas as pd
            from datetime import date,datetime

            time_now = datetime.now()
            current_time = time_now.strftime("%H:%M:%S")
            df_old = pd.read_excel("attendance.xlsx",sheet_name="attendance")
            dataframe=pd.DataFrame({'Name':[str(id)],
                                    'Date':[str(date.today())],
                                    'Time':[str(current_time)]
                                    })
            dataframe=df_old.append(dataframe)
            writer = pd.ExcelWriter("attendance.xlsx", engine='xlsxwriter')
            dataframe.to_excel(writer,sheet_name = "attendance", index=False)
            writer.save()
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(1) # Press 'ESC' for exiting video
    if k == 113:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()