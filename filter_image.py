import cv2
import numpy as np
import os
from filters import *


def place_img(img, x, y, img2):
    h, w, _ = img2.shape
    img[y:y+h, x:x+w] = img2


if __name__ == '__main__':
    # Read image
    img = cv2.imread(os.getcwd() + '/img/Mod4CT1.jpeg')
    orig_h, orig_w, orig_d = img.shape

    processed_img = np.zeros((orig_h*3+3, orig_w*4+4,  orig_d), np.uint8)

    for key, value in enumerate([3, 5, 7]):
        tmp = mean_filter(img, value)
        place_img(processed_img, 0, key*orig_h, tmp)
        tmp = median_filter(img, value)
        place_img(processed_img, orig_w, key*orig_h, tmp)
        tmp = gaussian_filter(img, value, 0)
        place_img(processed_img, orig_w*2, key*orig_h, tmp)
        tmp = gaussian_filter(img, value, 2)
        place_img(processed_img, orig_w*3, key*orig_h, tmp)

    h, w, d = processed_img.shape
    label_area_size = 100
    final_img = np.zeros(
        (h+label_area_size, w+label_area_size, orig_d), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key, value in enumerate(["Mean Filter", "Median Filter", "Gaussian Filter (0)", "Gaussian Filter (2)"]):
        cv2.putText(final_img, value,
                    (key*orig_w+label_area_size, int(label_area_size/2)),  font, .6, (255, 255, 255))
    for key, value in enumerate([3, 5, 7]):
        cv2.putText(final_img, "kernel: " + str(value),
                    (5, (key+1)*orig_h), font, .6, (255, 255, 255))

    place_img(final_img, label_area_size, label_area_size, processed_img)

    cv2.imwrite(os.getcwd() + '/img/Mod4CT1_processed.jpeg', final_img)
