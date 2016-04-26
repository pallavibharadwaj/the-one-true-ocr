import numpy
from extractor import get_feature_list, get_class_list
def generate_ground_data(image,copy, segments):
    feature_list= get_feature_list(image, segments)
    classes_list= get_class_list(copy, segments)
    with open("test.txt",'wb') as test :
        for char,feature in zip(classes_list,feature_list) :
                test.write("%s %s\n" %(chr(char), ' '.join(map(str,feature))))

def load_data_from_file():
    with open("test.txt") as char_file:
    	ncols = len(char_file.readline().split(' '))
    classes = numpy.loadtxt("test.txt", dtype = str, usecols = [0])
    features = numpy.loadtxt("test.txt", dtype = float, usecols = range(1,ncols))
    classes = [ord(x) for x in classes]
    features= numpy.asarray(features, dtype=numpy.float32 )
    classes= numpy.asarray( classes, dtype=numpy.float32 )
    return classes, features
