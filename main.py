import cv2
import numpy
import os
from extractor import get_feature_list, get_class_list
from preprocessor import preprocess, preprocess_with_display
from files import load_data_from_file, generate_ground_data

def read_image(image) :
	image = cv2.imread("%s" % image)
	return image

def files_creation(train_file) :
	img_path,extension=os.path.splitext("%s" % train_file) #splits the image path and the extension
	img_name= os.path.basename(img_path) #extracts the filename from an extension
	fp = open('%s.box' % img_path,'wb')	#create a .box file for the image
	txt_file = img_path + ".txt"    # .txt file that will be used to hold features of the image
	return txt_file

def train(train_images) :
	train_image_names = []
	image = []
	segments = []
	txt_files = []
	class_list = numpy.empty(shape=[0,])
	feature_list= numpy.empty(shape=[0,13])
	print train_images

	for image in train_images :
		print "\nCurrent image: ",image, "\n"
		txt_file = files_creation(image)
		image = read_image(image)	#reading training images
		copy=image.copy()
		image, segments = preprocess_with_display(image)
		#generate_ground_data(txt_file,image,copy,segments)
		classes,features = load_data_from_file(txt_file)
		dimension = classes.shape
		dimension1 = features.shape
		print "classes dimension: ", dimension
		print "features dimension: ", dimension1 
		class_list = numpy.concatenate((class_list , classes), axis=0)
		feature_list = numpy.concatenate((feature_list , features), axis=0)
		test_feature_list = get_feature_list(image, segments2)
		test_feature_list =  numpy.asarray( test_feature_list, dtype=numpy.float32 )
		
	class_list = numpy.asarray(class_list, dtype=numpy.float32)
	feature_list = numpy.asarray(feature_list,dtype=numpy.float32)
	print "total number of classes: ",len(class_list)
	print "total number of features: ",len(feature_list)
	return class_list, feature_list, test_feature_list

in_image = "data/alpha2.png"
input_image = read_image(in_image)	#input image to OCR
input_image, segments2 = preprocess(input_image) #preprocess of test image

train_images = "data/alpha.png","data/alpha2.png"
print train_images
class_list, feature_list, test_feature_list = train(train_images)

knn = cv2.ml.KNearest_create()
knn.train(feature_list,cv2.ml.ROW_SAMPLE, class_list)
retval, result_classes, neigh_resp, dists= knn.findNearest(test_feature_list, k= 1)
temp_classes = result_classes.tolist()
flattened = [chr(int(val)) for sublist in temp_classes for val in sublist]
print flattened

#f.close()
cv2.waitKey(0)
cv2.destroyAllWindows()
