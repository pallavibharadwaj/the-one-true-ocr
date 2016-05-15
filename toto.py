#!/usr/bin/env python
import argparse
import cv2
import numpy
from preprocessor import preprocess
from ocr import train, test, knnModel, SVMModel, format_spaces, test_accuracy
parser = argparse.ArgumentParser()

train_images = ["data/freeMono_train.png","data/freeSans_train.png","data/freeSerif_train.png","data/ubuntuCondensed_train.png","data/liberationSerif_train.png","data/timesNewRoman_train.png","data/giliusAdf_train.png","data/latinModernMonoLight_train.png","data/inconsolata_train.png"]
class_list, feature_list = train(train_images)

parser.add_argument("-m", "--model", default="knn", help="Decide model to be used")
parser.add_argument("image", help="Choose image to be used run OCR on")
parser.add_argument("-s","--space",default="10",help="Define space thresholds")
parser.add_argument("--accuracy",action="store_true",help="Turn on accuracy display")
args = parser.parse_args()
test_feature_list,spaces_list = test(args.image)
if (args.model == "knn") | (args.model == "KNN"):
    result = knnModel(feature_list, class_list, test_feature_list)
    print format_spaces(result,spaces_list,int(args.space))
    if(args.accuracy):
        test_accuracy(args.image,result)
elif (args.model == "SVM") | (args.model == "svm"):
    result = SVMModel(feature_list, class_list, test_feature_list)
    print format_spaces(result,spaces_list,int(args.space))
    if(args.accuracy):
        test_accuracy(args.image,result)
else:
    print "Error, model not defined"
