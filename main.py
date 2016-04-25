import cv2
import numpy
from extractor import find_eular_and_inner_segments, find_total_on_pixels, get_char
from extractor import extract_coordinate_based_features, get_feature_list, get_class_list
from segmentation import segments_to_numpy, draw_segments

#f=open("data.txt",'w')

image = cv2.imread('alpha.png')
image2 = cv2.imread('alpha2.png')
cv2.imshow('Display', image)
copy = image.copy()  # Keeping a non-greyscale copy of the image for later use
cv2.waitKey(0)
image = cv2.GaussianBlur(image, (5, 5), 0)  # Blurring the image
image2 = cv2.GaussianBlur(image2, (5,5), 0)
cv2.imshow('Display', image)
print("After Guassian blurring")
cv2.waitKey(0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grey scaling the image
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
cv2.imshow('Display', image)  # Displaying the image
print("After Greyscaling")
cv2.waitKey(0)
image = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 10)
image2 = cv2.adaptiveThreshold(
    image2,
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
im22, contours2, hierarchy2 = cv2.findContours(
    image2,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE)
''' Finding the contours in the image'''

image.fill(255)
image2.fill(255)
cv2.drawContours(image, contours, -1, (0, 0, 255))
cv2.drawContours(image2, contours2, -1, (0, 0, 255))
''' Drawing the contours on the image before displaying'''
cv2.imshow('Display', image)
print("After detecting Contours")
cv2.waitKey(0)
contours.reverse()
contours2.reverse()
segments = segments_to_numpy([cv2.boundingRect(c) for c in contours])
segments2 = segments_to_numpy([cv2.boundingRect(c) for c in contours2])
''' Convert each segment to x,y,w,h (4-element tuple for numpy)'''
copy_for_grounding = copy.copy()
draw_segments(copy, segments)
'''Draw the segments on the copy image (cant add color to greyscaled image)'''
cv2.imshow('Display', copy)
print("After Segmentation")
cv2.waitKey(0)
# Finding the outer bounding box to find segments within a segment
feature_list2 = []
classes_list2 = []
knn = cv2.ml.KNearest_create()
# To get the training data from image
    # f.write(' '.join(map(str, final_data))+"\n")

feature_list= get_feature_list(image, segments)
classes_list= get_class_list(copy_for_grounding, segments)
feature_list2 = get_feature_list(image2,segments2)

features= numpy.asarray( feature_list, dtype=numpy.float32 )
classes= numpy.asarray( classes_list, dtype=numpy.float32 )
features2 =  numpy.asarray( feature_list2, dtype=numpy.float32 )

knn.train(features,cv2.ml.ROW_SAMPLE, classes)
retval, result_classes, neigh_resp, dists= knn.findNearest(features2, k= 1)
temp_classes = result_classes.tolist()

flattened = [chr(int(val)) for sublist in temp_classes for val in sublist]
print flattened

#f.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
