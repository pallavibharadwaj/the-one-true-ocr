import os
import cv2
import numpy
from extractor import get_feature_list, get_class_list
from preprocessor import preprocess


def generate_ground_data(image_path):
    image, img_txt = read_image(image_path)
    copy = image.copy()
    image, segments, euler_list, central_x, central_y = preprocess(image)
    feature_list = get_feature_list(
        image, segments, euler_list, central_x, central_y)[0]
    classes_list = get_class_list(copy, segments)
    with open("%s" % img_txt, 'wb') as test:
        for char, feature in zip(classes_list, feature_list):
            test.write("%s %s\n" % (chr(char), ' '.join(map(str, feature))))


def load_data_from_file(img_txt):
    with open("%s" % img_txt) as char_file:
        ncols = len(char_file.readline().split(' '))
    classes = numpy.loadtxt("%s" % img_txt, dtype=str, usecols=[0])
    features = numpy.loadtxt(
        "%s" % img_txt, dtype=float, usecols=range(1, ncols))
    classes = [ord(x) for x in classes]
    features = numpy.asarray(features, dtype=numpy.float32)
    classes = numpy.asarray(classes, dtype=numpy.float32)
    return classes, features


def read_image(image):
    # splits the image path and the extension
    img_path = os.path.splitext("%s" % image)[0]
    # .txt file that will be used to hold features of the image
    txt_file = img_path + ".txt"
    image = cv2.imread("%s" % image)
    return image, txt_file
