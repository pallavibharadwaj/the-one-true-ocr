import cv2
import numpy
import os
from extractor import get_feature_list, get_class_list
from preprocessor import preprocess, preprocess_with_display
from files import load_data_from_file, generate_ground_data
image_1="data/alpha.png"
image_2="data/alpha2.png"

def read_image(image_1,image_2) :
	image = cv2.imread("%s" % image_1)
	image2 = cv2.imread("%s" %image_2)
	return image,image2

image,image2 = read_image(image_1,image_2)
img,extension=os.path.splitext("%s" %image_1) #splits the image path and the extension
#print img #data/alpha
in_file = os.path.basename(img) #extracts the filename from an extension
#print in_file #alpha
file = open('%s.box' % in_file,'a')
copy=image.copy()
image, segments = preprocess_with_display(image)
image2, segments2 = preprocess(image2)
knn = cv2.ml.KNearest_create()
feature_list2 = get_feature_list(image, segments2)
in_file = img + ".txt"    #Ex:data/alpha.txt
#print in_file
generate_ground_data(in_file,image,copy,segments)
classes,features = load_data_from_file(in_file)
features2 =  numpy.asarray( feature_list2, dtype=numpy.float32 )
knn.train(features,cv2.ml.ROW_SAMPLE, classes)
retval, result_classes, neigh_resp, dists= knn.findNearest(features2, k= 1)
temp_classes = result_classes.tolist()
flattened = [chr(int(val)) for sublist in temp_classes for val in sublist]
print flattened

#f.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
