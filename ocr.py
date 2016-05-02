import numpy
from preprocessor import preprocess
import cv2
from files import load_data_from_file
import os
from extractor import get_feature_list

def read_image(image) :
	img_path,extension=os.path.splitext("%s" % image) #splits the image path and the extension
	img_name= os.path.basename(img_path) #extracts the filename from an extension
	txt_file = img_path + ".txt"    # .txt file that will be used to hold features of the image
	image = cv2.imread("%s" % image)
	return image, txt_file


def train(train_images) :
	train_image_names = []
	image = []
	segments = []
	txt_files = []
	class_list = numpy.empty(shape=[0,])
	feature_list= numpy.empty(shape=[0,13])

	for image in train_images :
		image , txt_file = read_image(image)	#reading training images
		copy=image.copy()
		image, segments = preprocess(image)
		classes,features = load_data_from_file(txt_file)
		dimension = classes.shape
		dimension1 = features.shape
		class_list = numpy.concatenate((class_list , classes), axis=0)
		feature_list = numpy.concatenate((feature_list , features), axis=0)

	class_list = numpy.asarray(class_list, dtype=numpy.float32)
	feature_list = numpy.asarray(feature_list,dtype=numpy.float32)
	return class_list, feature_list

def test(image):
	input_image,test_txt_file = read_image(image)	#input image to OCR
	copy = input_image.copy()
	input_image, segments2 = preprocess(input_image) #preprocess of test image
	test_feature_list = get_feature_list(input_image, segments2)
	test_feature_list =  numpy.asarray( test_feature_list, dtype=numpy.float32 )
	cv2.imshow('Test Image', copy)
	return test_feature_list

def knnModel(training_features, training_classes, test_features):
	knn = cv2.ml.KNearest_create()
	knn.train(training_features, cv2.ml.ROW_SAMPLE, training_classes)
	retval, result_classes, neigh_resp, dists = knn.findNearest(test_features, k = 1)
	temp_classes = result_classes.tolist()
	flattened = [chr(int(val)) for sublist in temp_classes for val in sublist]
	return flattened
