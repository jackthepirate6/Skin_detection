import cv2
import numpy as np
import time

import tensorflow as tf
from skimage.measure import compare_ssim as ssim


def cal_per(img):
    total = img.shape[0]*img.shape[1]
    white = 0
    for a in img:
        for b in a:
            if(b!=0):
                white = white + 1   
            
    black = total - white
    return black/total


def lowlight_test(lowlight_enhance,test_low_data):
    test_high_data = []
    test_low_data=np.array(test_low_data, dtype="float32") / 255.0
    img=lowlight_enhance.inference(test_low_data, test_high_data)
    img=np.array(img[...,::-1])
    print("low light here")
    return(img)


def detect(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(img_HSV, (0, 40, 0), (25,255,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 138, 67), (255,173,133))
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


    global_mask=cv2.bitwise_or(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)

    percent_1 = cal_per(HSV_result)
    percent_2 = cal_per(YCrCb_result)
    percent_3 = cal_per(global_result)
    avg_per = (percent_1 + percent_2 + percent_3)/3
   
    if(avg_per > 0.095):
        return True
    else:
        return False
        
