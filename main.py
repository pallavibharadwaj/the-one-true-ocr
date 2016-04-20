import cv2
import numpy
from extractor import find_eular, find_total_on_pixels
from extractor import extract_coordinate_based_features
SEGMENTS_DIRECTION = 0  # vertical axis in numpy


def segments_to_numpy(segments):
    '''given a list of 4-element tuples, transforms it into a numpy array'''
    segments = numpy.array(segments, dtype=numpy.uint16, ndmin=2)
    # each segment in a row
    if SEGMENTS_DIRECTION != 0:
        numpy.transpose(segments)
    return segments


def draw_segments(image, segments, color=(255, 0, 0), line_width=1):
        '''draws segments on image'''
        for segment in segments:
            x, y, w, h = segment
            cv2.rectangle(image, (x, y), (x+w, y+h), color, line_width)

image = cv2.imread('alpha.png')
cv2.imshow('Display', image)
copy = image.copy()  # Keeping a non-greyscale copy of the image for later use
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
segments = segments_to_numpy([cv2.boundingRect(c) for c in contours])
''' Convert each segment to x,y,w,h (4-element tuple for numpy)'''
draw_segments(copy, segments)
'''Draw the segments on the copy image (cant add color to greyscaled image)'''
cv2.imshow('Display', copy)
print("After Segmentation")
cv2.waitKey(0)
# Finding the outer bounding box to find segments within a segment
xmin, ymin, wmin, hmin = numpy.amin(segments, axis=0)
xmax, ymax, wmax, hmax = numpy.amax(segments, axis=0)
print "number of segments:{}".format(len(segments))
# number of segments for reference and confirmation
print "Feature Data: Euler Number, Horizontal Mean, Vertical Mean"
feature_list = []
for each_segment in segments:
    if (each_segment == [xmin, ymin, wmax, hmax]).all():
        # Skipping the large segment
        continue
    euler_number = find_eular(each_segment, segments)
    on_pixel = find_total_on_pixels(image, each_segment)
    coordinate_features = extract_coordinate_based_features(
        image,
        on_pixel,
        each_segment)
    # Converting all the feature data into a tuple
    feature_list.append([euler_number, on_pixel]+coordinate_features)
print feature_list
cv2.waitKey(0)
cv2.destroyAllWindows()
