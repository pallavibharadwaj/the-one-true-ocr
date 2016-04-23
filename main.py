import cv2
import numpy
from extractor import find_eular, find_total_on_pixels, get_char
from extractor import extract_coordinate_based_features, inner_segment
from segmentation import segments_to_numpy, draw_segments

f=open("data.txt",'w')

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
xmin, ymin, wmin, hmin = numpy.amin(segments, axis=0)
xmax, ymax, wmax, hmax = numpy.amax(segments, axis=0)
feature_list = []
classes_list = []
feature_list2 = []
classes_list2 = []
knn = cv2.ml.KNearest_create()
for each_segment in segments:
    if (each_segment == [xmin, ymin, wmax, hmax]).all():
        # Skipping the large segment
        continue
    euler_number = find_eular(each_segment, segments)
#add code to remove inner segments
inner_segments= inner_segment()

flag=0
for each_segment in segments:
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
    final_data = segment_features
    feature_list.append(final_data)
    classes_list.append(ord(char_data))
    # f.write(' '.join(map(str, final_data))+"\n")


xmin1, ymin1, wmin1, hmin1 = numpy.amin(segments2, axis=0)
xmax1, ymax1, wmax1, hmax1 = numpy.amax(segments2, axis=0)


for each_segment in segments2:
    if (each_segment == [xmin1, ymin1, wmax1, hmax1]).all():
        # Skipping the large segment
        continue
    euler_number = find_eular(each_segment, segments2)
#add code to remove inner segments
inner_segments2= inner_segment()

flag=0

for each_segment in segments2:
    bad_flag=0
    x,y,w,h=each_segment
    if (each_segment == [xmin1, ymin1, wmax1, hmax1]).all():
        print "large segment"
        # Skipping the large segment
        continue
    for inner in inner_segments2:
        x1,y1,w1,h1=inner
        if ([x,y,w,h]==[x1,y1,w1,h1]):
            flag=flag+1
            bad_flag=1
            continue
    if(bad_flag==1):
        print "bad_flag"
        continue
    print "in the loop ,ike a BOSS"
    euler_number = find_eular(each_segment, segments2)
    on_pixel = find_total_on_pixels(image2, each_segment)
    #cv2.waitKey(0)
    coordinate_features = extract_coordinate_based_features(
        image2,
        on_pixel,
        each_segment)
    # Converting all the feature data into a tuple
    segment_features = [euler_number, on_pixel]+coordinate_features
    #char_data = get_char(copy_for_grounding, each_segment, segment_features)
    final_data = segment_features
    feature_list2.append(final_data)
    #classes_list2.append(ord(char_data))

#print "flist1",feature_list
print "flist2",feature_list2

print "classes",classes_list
#print "Classes2",classes_list2

features= numpy.asarray( feature_list, dtype=numpy.float32 )
classes= numpy.asarray( classes_list, dtype=numpy.float32 )
features2 =  numpy.asarray( feature_list2, dtype=numpy.float32 )
#print('haha',features)
#print('help',features2)
knn.train(features,cv2.ml.ROW_SAMPLE, classes)
retval, result_classes, neigh_resp, dists= knn.findNearest(features2, k= 1)
temp_classes = result_classes.tolist()
flattened = [chr(int(val)) for sublist in temp_classes for val in sublist]
print flattened

    # draw_individual_segment(copy_for_grounding, each_segment)
# print feature_list
f.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
