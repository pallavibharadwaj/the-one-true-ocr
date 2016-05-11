import cv2
import numpy
from preprocessor import preprocess
from ocr import train, test, knnModel, SvmModel
from extractor import get_feature_list
from files import generate_ground_data

def format_spaces(result,spaces_list,space_value):
	newline_flag=1
	final_string = ""
	if(len(spaces_list)<=len(result)):
		iterate_value = spaces_list
		# Handles out of bounds in case of spaces < results
	else:
		iterate_value = result
	for i in range(len(iterate_value)):
		if newline_flag==1:
			result_str= str(result[i])
			newline_flag=0
			continue
		if spaces_list[i]>0 and spaces_list[i]<=space_value :
			result_str+=str(result[i])
			newline_flag=0
		elif spaces_list[i]>space_value:
			result_str+="  "+str(result[i])
			newline_flag=0
		elif spaces_list[i]<0:
			result_str+=str(result[i]) # Add last char of each line
			newline_flag=1
			final_string+='\n'+ result_str

	if(len(iterate_value)<len(result)):
		final_string+='\n'+result_str
		# Add last lines to images with lower spaces
	final_string+=result[-1] # Last Character added to result
	return final_string

in_image = "data/alpha.png"
test_feature_list,spaces_list= test(in_image)

train_images = ["data/alpha.png"]
class_list, feature_list = train(train_images)
result = knnModel(feature_list,class_list,test_feature_list)
result2 = SvmModel(feature_list,class_list,test_feature_list)
print result
print result2
print "Formatted results"
print(format_spaces(result,spaces_list,10))
print(format_spaces(result2,spaces_list,10))
#generate_ground_data("data/Agenda.png")
cv2.waitKey(0)
cv2.destroyAllWindows()
