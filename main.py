import cv2
import numpy
from extractor import get_feature_list, get_class_list
from preprocessor import preprocess, preprocess_with_display
from files import load_data_from_file, generate_ground_data

image = cv2.imread('alpha.png')
image2 = cv2.imread('alpha2.png')
copy=image.copy()
image, segments = preprocess_with_display(image)
image2, segments2 = preprocess(image2)
knn = cv2.ml.KNearest_create()
feature_list2 = get_feature_list(image, segments2)
#generate_ground_data(image,copy,segments)
classes, features = load_data_from_file()
features2 =  numpy.asarray( feature_list2, dtype=numpy.float32 )
knn.train(features,cv2.ml.ROW_SAMPLE, classes)
retval, result_classes, neigh_resp, dists= knn.findNearest(features2, k= 1)
temp_classes = result_classes.tolist()
flattened = [chr(int(val)) for sublist in temp_classes for val in sublist]
print flattened

#f.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
