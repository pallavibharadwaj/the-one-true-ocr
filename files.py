import cv2
import numpy
from extractor import get_feature_list, get_class_list
import os
from preprocessor import preprocess,preprocess_with_display

def generate_ground_data(image_path):
    image ,img_txt = read_image(image_path)
    copy = image.copy()
    image,segments,euler_list,central_x,central_y = preprocess_with_display(image)
    feature_list,spaces_list= get_feature_list(image, segments , euler_list,central_x,central_y)
    classes_list = get_class_list(copy, segments , euler_list)
    with open("%s" % img_txt, 'wb') as test:
        for char,feature in zip(classes_list, feature_list):
            test.write("%s %s\n" % (chr(char), ' '.join(map(str, feature))))


def load_data_from_file(img_txt):
    with open("%s" % img_txt) as char_file:
        ncols = len(char_file.readline().split(' '))
    classes = numpy.loadtxt("%s" % img_txt, dtype = str, usecols = [0])
    features = numpy.loadtxt("%s" %img_txt, dtype = float, usecols = range(1,ncols))
    classes = [ord(x) for x in classes]
    features = numpy.asarray(features, dtype=numpy.float32)
    classes = numpy.asarray(classes, dtype=numpy.float32)
    return classes, features

def read_image(image) :
	img_path,extension=os.path.splitext("%s" % image) #splits the image path and the extension
	img_name= os.path.basename(img_path) #extracts the filename from an extension
	txt_file = img_path + ".txt"    # .txt file that will be used to hold features of the image
	image = cv2.imread("%s" % image)
	return image, txt_file
