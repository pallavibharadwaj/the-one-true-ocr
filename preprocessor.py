import cv2
from segmentation import segments_to_numpy, draw_segments,segment_blocks
def preprocess_with_display(image):
    copy = image.copy()
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Blurring the image
    cv2.imshow('Display', image)
    print("After Guassian blurring")
    cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grey scaling the image
    cv2.imshow('Display', image)  # Displaying the image
    print("After Greyscaling")
    cv2.waitKey(0)
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 10)
    ''' Thresholding the image (Converts greyscale to binary),
        using adaptive threshold for best results '''
    cv2.imshow('Display', image)
    im2, contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
    ''' Finding the contours in the image'''
    image.fill(255)
    cv2.drawContours(image, contours, -1, (0, 0, 255))
    ''' Drawing the contours on the image before displaying'''
    cv2.imshow('Display', image)
    print("After detecting Contours")
    cv2.waitKey(0)
    contours.reverse()
    segments = segments_to_numpy([cv2.boundingRect(c) for c in contours])
    segments=segment_blocks(segments)
    draw_segments(copy, segments)
    '''Draw the segments on the copy image (cant add color to greyscaled image)'''
    cv2.imshow('Display', copy)
    print("After Segmentation")
    cv2.waitKey(0)
    return image, segments


def preprocess(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Blurring the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grey scaling the image
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 10)
    ''' Thresholding the image (Converts greyscale to binary),
        using adaptive threshold for best results '''
    im2, contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
    ''' Finding the contours in the image'''
    image.fill(255)
    cv2.drawContours(image, contours, -1, (0, 0, 255))
    ''' Drawing the contours on the image before displaying'''
    contours.reverse()
    segments = segments_to_numpy([cv2.boundingRect(c) for c in contours])
    segments=segment_blocks(segments)
    return image, segments
