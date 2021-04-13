import cv2
import numpy as np
from keras.models import load_model

model=load_model('Model/model.h5')
image=cv2.imread('Test_Images/apple.png')

image_copy=image.copy()
img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)
ret, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

ctrs,hier=cv2.findContours(img_thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects=[cv2.boundingRect(ctr) for ctr in ctrs]

for rect in rects:
    x,y,w,h=rect[0],rect[1],rect[2],rect[3]
    X=x+w+15
    Y=y+h+15
    cv2.rectangle(image,(x-10,y-10),(X,Y),(255,0,0),2)

    roi=img_thresh[y-20:Y,x-20:X]
    #cv2.imshow('Roi',roi)

    roi = cv2.resize(roi, (28, 28),interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi,(3,3))
    roi = np.reshape(roi, (1, 28, 28, 1))


    letters_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'
        , 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O'
        , 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
                    22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

    pred = model.predict(roi)
    num = np.argmax(pred)

    probVal = np.amax(pred)
    predict = str(letters_dict[num])

    cv2.putText(image, "Letter: " + str(predict), (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, "Accuracy: " + str(probVal), (x, Y+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('Image',image)
cv2.waitKey()