import cv2
import numpy as np
from keras.models import load_model

letters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G'
    ,7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O'
    ,15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',
                22:'W',23:'X', 24:'Y',25:'Z'}

threshold=0.65
model=load_model('Model/model.h5')
camera=cv2.VideoCapture(0)

def preProcessing(img):
    img_copy = img.copy()
    img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    return img_final

def main():
    while True:
        ret,frame=camera.read()
        x, y, w, h = (0, 0, 280, 300)
        cut = frame[y:y + h, x:x + w]

        img = preProcessing(cut)
        pred = model.predict(img)
        num = np.argmax(pred)

        probVal = np.amax(pred)
        predict = str(letters_dict[num])
        print("Letter: ", predict)
        print("Accuracy: ", probVal)

        if probVal > threshold:
            cv2.putText(frame, "Digit: " + str(predict), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Accuracy: " + str(probVal), (5, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Cut",cut)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()