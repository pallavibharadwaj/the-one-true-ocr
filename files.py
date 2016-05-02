import cv2
import numpy
from extractor import get_feature_list, get_class_list


def generate_ground_data(img_txt,image,copy, segments):
    feature_list= get_feature_list(image, segments)
    classes_list = get_class_list(copy, segments)
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
