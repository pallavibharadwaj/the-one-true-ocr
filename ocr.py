import numpy
from preprocessor import preprocess
import cv2
from files import load_data_from_file, read_image
from extractor import get_feature_list
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import cPickle

def train(train_images):
	train_image_names = []
	image = []
	segments = []
	txt_files = []
	class_list = numpy.empty(shape=[0,])
	feature_list= numpy.empty(shape=[0,13])

	for image in train_images:
		image , txt_file = read_image(image)	#reading training images
		copy=image.copy()
		image, segments , eular_list= preprocess(image)
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
	input_image, segments2 , eular_list = preprocess(input_image) #preprocess of test image
	test_feature_list = get_feature_list(input_image, segments2,eular_list)
	test_feature_list =  numpy.asarray( test_feature_list, dtype=numpy.float32 )
	cv2.imshow('Test Image', copy)
	return test_feature_list

def knnModel(training_features, training_classes, test_features):
	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(training_features, training_classes)
	#f1  = open('data/KNNmodel.pkl', 'wb')
	#cPickle.dump(knn , f1 , protocol = cPickle.HIGHEST_PROTOCOL)
	#f1.close()
	#knn2 = cPickle.load(open('data/KNNmodel.pkl' , 'rb'))
	results = knn.predict(test_features)
	results = [chr(int(val)) for val in results]
	return results

def SVMModel(training_features, training_classes, test_features):
	svm = SVC()
	svm.fit(training_features,training_classes)
	#f2  = open('data/SVMmodel.pkl', 'wb')
	#cPickle.dump(svm , f2 , protocol = cPickle.HIGHEST_PROTOCOL)
	#f2.close()
	#svm2 = cPickle.load(open('data/SVMmodel.pkl' , 'rb'))
	result_classes = svm.predict(test_features)
	results = [chr(int(val)) for val in result_classes]
	return results
