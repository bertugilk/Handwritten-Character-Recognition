import cv2
import numpy as np
from keras.models import load_model

model=load_model('Model/model.h5')
image=cv2.imread('Test_Images/a.png')

def preProcessing(img):
    img_copy = img.copy()
    img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    return img_final

letters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'
    ,7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O'
    ,15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',
                22:'W',23:'X', 24:'Y',25:'Z'}

img=preProcessing(image)
pred=model.predict(img)
num=np.argmax(pred)

probVal = np.amax(pred)
predict=str(letters_dict[num])

print("Letter: ",predict)
print("Accuracy: ",probVal)