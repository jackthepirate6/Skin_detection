import cv2
import numpy as np
import easygui
from skimage.measure import compare_ssim as ssim
import time



path = easygui.fileopenbox()
img = cv2.imread(path)
if(max(img.shape[0],img.shape[1]) > 1000):
    img = cv2.resize(img,(100,100),interpolation = cv2.INTER_LANCZOS4)

start = time.time()
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
HSV_mask = cv2.inRange(img_HSV, (0, 40, 0), (25,255,255)) 
HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
YCrCb_mask = cv2.inRange(img_YCrCb, (0, 138, 67), (255,173,133))

YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
global_mask=cv2.medianBlur(global_mask,3)
global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


HSV_result = cv2.bitwise_not(HSV_mask)
YCrCb_result = cv2.bitwise_not(YCrCb_mask)
global_result=cv2.bitwise_not(global_mask)


h = cv2.cvtColor(HSV_result, cv2.COLOR_GRAY2BGR)
y = cv2.cvtColor(YCrCb_result, cv2.COLOR_GRAY2BGR)
g = cv2.cvtColor(global_result, cv2.COLOR_GRAY2BGR)

h = cv2.cvtColor(h, cv2.COLOR_BGR2GRAY)
y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)
g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
s1 = ssim(h,y)
s2 = ssim(y,g)
s3 = ssim(h,g)
end = time.time()
print("this is time required ",end-start)


if(max(img.shape[0],img.shape[1]) > 200):
    threshold = 0.75
else:
    threshold = 0.56
    
print("this is threshold ",threshold)
if(threshold == 0.75):
    if((s1 > threshold or s2 > threshold) and abs(s1 - s2) <= 0.1):
        print("non face ")
    else:
        print("face")

if(threshold == 0.56):
    if(s1 > threshold or s2 > threshold):
        print("non face")
    else:
        print("face")

#if((s1 > threshold or s2 > threshold) and abs(s1 - s2) <= 0.1 and i == 0):
#    print("non face, ")
#else:
#    print("face ")
