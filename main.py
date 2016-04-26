import cv2
import numpy
from extractor import find_eular_and_inner_segments, find_total_on_pixels, get_char
from extractor import extract_coordinate_based_features, get_feature_list, get_class_list
from segmentation import segments_to_numpy, draw_segments
from preprocessor import preprocess, preprocess_with_display
#f=open("data.txt",'w')

image = cv2.imread('alpha.png')
image2 = cv2.imread('alpha.png')
copy=image.copy()
image, segments = preprocess_with_display(image)
image2, segments2 = preprocess(image2)
''' Convert each segment to x,y,w,h (4-element tuple for numpy)'''
copy_for_grounding = copy.copy()
cv2.waitKey(0)
# Finding the outer bounding box to find segments within a segment
feature_list2 = []
classes_list2 = []
knn = cv2.ml.KNearest_create()
# To get the training data from image
    # f.write(' '.join(map(str, final_data))+"\n")

feature_list= get_feature_list(image, segments)
classes_list= get_class_list(copy, segments)
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
