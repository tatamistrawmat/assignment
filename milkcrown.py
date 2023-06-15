import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def remove_objects(img, lower_size=None, upper_size=None):
    # find all objects
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    sizes = stats[1:, -1]
    _img = np.zeros((labels.shape))

    # process all objects, label=0 is background, objects are started from 1
    for i in range(1, nlabels):

        # remove small objects
        if (lower_size is not None) and (upper_size is not None):
            if lower_size < sizes[i - 1] and sizes[i - 1] < upper_size:
                _img[labels == i] = 255

        elif (lower_size is not None) and (upper_size is None):
            if lower_size < sizes[i - 1]:
                _img[labels == i] = 255

        elif (lower_size is None) and (upper_size is not None):
            if sizes[i - 1] < upper_size:
                _img[labels == i] = 255

    return _img

def cut_water_crown():
    imc = cv2.imread("milkdrop.bmp")
    img = cv2.cvtColor( imc , cv2.COLOR_BGR2GRAY)
    
    # 閾値の設定
    threshold = 100

    # 二値化(閾値100を超えた画素を255にする。)
    ret, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_OTSU) #cv2.THRESH_BINARY)
   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations = 1)
    img_bin = remove_objects(img_bin, lower_size=5000)
    img_bin = cv2.dilate(img_bin, kernel_d, iterations=2) # 膨張（Dilation）- dilate()
    img_bin = cv2.erode(img_bin, kernel_e, iterations=2)  # 収縮（Erison）- erode()

    img_bin = img_bin.astype(np.uint8)
    for i in range(len(img_bin)):
        for j in range(len(img_bin[0])):
            if img_bin[i,j] == 0:
                imc[i, j] = [0,0,0]

    cv2.imshow('image',imc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cut_water_crown()
