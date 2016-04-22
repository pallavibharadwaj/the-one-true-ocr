import cv2
import numpy
from extractor import find_eular, find_total_on_pixels, get_char
from extractor import extract_coordinate_based_features, inner_segment
from segmentation import segments_to_numpy, draw_segments

f=open("data.txt",'w')

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
contours.reverse()
segments = segments_to_numpy([cv2.boundingRect(c) for c in contours])
''' Convert each segment to x,y,w,h (4-element tuple for numpy)'''
copy_for_grounding = copy.copy()
draw_segments(copy, segments)
'''Draw the segments on the copy image (cant add color to greyscaled image)'''
cv2.imshow('Display', copy)
print("After Segmentation")
cv2.waitKey(0)
# Finding the outer bounding box to find segments within a segment
xmin, ymin, wmin, hmin = numpy.amin(segments, axis=0)
xmax, ymax, wmax, hmax = numpy.amax(segments, axis=0)
feature_list = []

for each_segment in segments:
    if (each_segment == [xmin, ymin, wmax, hmax]).all():
        # Skipping the large segment
        continue
    euler_number = find_eular(each_segment, segments)
#add code to remove inner segments
inner_segments= inner_segment()

flag=0
for each_segment in segments:
    #if (each_segment == [xmin, ymin, wmax, hmax]).all():
    #    # Skipping the large segment
    #    continue
    #euler_number = find_eular(each_segment, segments)
    #add code to remove inner segments
    #inner_segments= inner_segment()
    #print inner_segments
    #print type(inner_segments)
    bad_flag=0
    x,y,w,h=each_segment
    for inner in inner_segments:
        x1,y1,w1,h1=inner
        if ([x,y,w,h]==[x1,y1,w1,h1]):
            flag=flag+1
            bad_flag=1
            continue
    if(bad_flag==1):
        continue
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
    segment_features = [euler_number, on_pixel]+coordinate_features
    char_data = get_char(copy_for_grounding, each_segment, segment_features)
    final_data = [char_data] + segment_features
    f.write(' '.join(map(str, final_data))+"\n")
    print(final_data)
print flag
    # draw_individual_segment(copy_for_grounding, each_segment)
# print feature_list
f.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
