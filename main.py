import cv2
import numpy
from preprocessor import preprocess
from ocr import train, test, knnModel, SVMModel, format_spaces
from extractor import get_feature_list
from files import generate_ground_data


in_image = "data/alpha.png"
test_feature_list,spaces_list= test(in_image)

train_images = ["data/alpha.png"]
class_list, feature_list = train(train_images)
result = knnModel(feature_list,class_list,test_feature_list)
result2 = SVMModel(feature_list,class_list,test_feature_list)
print result
print result2
print "Formatted results"
print(format_spaces(result,spaces_list,10))
print(format_spaces(result2,spaces_list,10))
#generate_ground_data("data/Agenda.png")
cv2.waitKey(0)
cv2.destroyAllWindows()
