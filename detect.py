import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img", type=str)

args = parser.parse_args()
img = args.img

ccd=cv2.CascadeClassifier('cascade.xml')
img= cv2.imread(img)
# cv2.imshow('PROCESSED IMAGE', img)
resized = cv2.resize(img,(400,200))
gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
dtc=ccd.detectMultiScale(gray,1.5,5)
for(x,y,w,h) in dtc:
    resized=cv2.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('img',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()