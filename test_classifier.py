import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
import cv2
from tensorflow.keras.models import load_model

model = load_model('classifier.h5')

#Predicting with the test data
y_test=pd.read_csv("./Test.csv")
paths=y_test['Path']
y_test=y_test['ClassId'].values
num_test_images=len(paths)

X_test = np.zeros((num_test_images, 32, 32, 3), dtype=np.float32)

for i in range(num_test_images):
    img = cv2.imread(paths[i])
    img = cv2.resize(img, (32,32), cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(4,4))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    X_test[i] = bgr/255.0
    print('\r'+'['+str(i+1)+'/'+str(num_test_images)+']',end="")

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print('Accuracy:',str(accuracy_score(y_test, y_pred)*100)+'%')