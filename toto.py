#!/usr/bin/env python
import argparse
import cv2
import numpy
from preprocessor import preprocess
from ocr import train, test, knnModel, SVMModel
parser = argparse.ArgumentParser()
# in_image = "data/alpha2.png"
# test_feature_list = test(in_image)
train_images = ["data/alpha.png"]
class_list, feature_list = train(train_images)

parser.add_argument("-m", "--model", default="knn", help="Decide model to be used")
parser.add_argument("image", help="Choose image to be used run OCR on")
args = parser.parse_args()
test_feature_list,_ = test(args.image)
if (args.model == "knn") | (args.model == "KNN"):
    result = knnModel(feature_list, class_list, test_feature_list)
    print result
elif (args.model == "SVM") | (args.model == "svm"):
    result = SVMModel(feature_list, class_list, test_feature_list)
    print result
else:
    print "Error, model not defined"
