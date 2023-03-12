# -------------------------- LBPH - Local Binary Pattern Histograms ------------------------
# Local Binary Pattern (LBP) is a simple yet very 
# efficient texture operator which labels the pixels 
# of an image by thresholding the neighborhood of each
# pixel and considers the result as a binary number.
# https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b

import os
from PIL import Image
import numpy as np
import cv2
import ctypes  # An included library with Python install.   


def train_classifier(data_dir):
    path=[ os.path.join(data_dir, file) for file in os.listdir(data_dir) ]
    faces=[]
    ids=[]

    for image in path:
        img=Image.open(image).convert("L")  #.convert("L") is to get value in grayscale

        # next convert into grid using numpy
        imageNP=np.array(img,'uint8')
        id=int(image.split(".")[1])

        faces.append(imageNP)
        ids.append(id)
        cv2.imshow("Training...",imageNP)
        cv2.waitKey(1)
    ids=np.array(ids)

    # ======================= Train and Saving the Classifier ===================

    # Training 
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    # Saving
    clf.write("attSysClassifier.xml")
    cv2.destroyAllWindows()
    ctypes.windll.user32.MessageBoxW(0, "Training Dataset Completed!!", "Result", 1)

train_classifier("imgs/")