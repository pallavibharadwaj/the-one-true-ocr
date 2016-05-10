import cv2
import numpy
from preprocessor import preprocess
from ocr import train, test, knnModel, SvmModel
from extractor import get_feature_list
from files import generate_ground_data

in_image = "data/alpha.png"
test_feature_list,spaces_list= test(in_image)

train_images = ["data/alpha.png"]
class_list, feature_list = train(train_images)
result = knnModel(feature_list,class_list,test_feature_list)
result2 = SvmModel(feature_list,class_list,test_feature_list)
print result

print "Formatted result"

newline_flag=1
for i in range(len(result)):
	if newline_flag==1:
		result_str=str(result[i])
		newline_flag=0
		continue
	if spaces_list[i]>0 and spaces_list[i]<=10 :
		result_str=result_str+str(result[i])
		newline_flag=0
	elif spaces_list[i]>10:
		result_str=result_str+"  "+str(result[i])
		newline_flag=0
	elif spaces_list[i]<0:
		newline_flag=1
		print result_str

print result2

print "Formatted result"

newline_flag=1
for i in range(len(result2)):
	if newline_flag==1:
		result_str2=str(result2[i])
		newline_flag=0
		continue
	if spaces_list[i]>0 and spaces_list[i]<=10 :
		result_str2=result_str2+str(result2[i])
		newline_flag=0
	elif spaces_list[i]>10:
		result_str2=result_str2+"  "+str(result2[i])
		newline_flag=0
	elif spaces_list[i]<0:
		newline_flag=1
		print result_str2

#generate_ground_data("data/Agenda.png")
cv2.waitKey(0)
cv2.destroyAllWindows()
