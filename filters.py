import cv2
import numpy as np


def mean_filter(img, kernel_size):
    # Create a mean filter
    kernel = np.ones((kernel_size, kernel_size),
                     np.float32) / (kernel_size ** 2)
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def median_filter(img, kernel_size):
    # Create a median filter
    dst = cv2.medianBlur(img, kernel_size)
    return dst


def gaussian_filter(img, kernel_size, sigma):
    # Create a gaussian filter
    dst = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    return dst
