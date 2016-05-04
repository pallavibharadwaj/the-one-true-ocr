import cv2
import numpy
from preprocessor import preprocess
from ocr import train, test, knnModel, SVMModel
from extractor import get_feature_list
from files import generate_ground_data

in_image = "data/alpha2.png"
test_feature_list = test(in_image)

train_images = "data/alpha.png", "data/alpha2.png"
class_list, feature_list = train(train_images)
generate_ground_data("data/alpha.png")
result = knnModel(feature_list,class_list,test_feature_list)
result2 = SVMModel(feature_list,class_list,test_feature_list)
print result
print result2
cv2.waitKey(0)
cv2.destroyAllWindows()
