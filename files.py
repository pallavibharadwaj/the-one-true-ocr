import numpy
from extractor import get_feature_list, get_class_list

def generate_ground_data(in_file,image,copy, segments):
    #print in_file
    feature_list= get_feature_list(image, segments)
    classes_list= get_class_list(copy, segments)
    with open("%s" % in_file,'wb') as test :
        for char,feature in zip(classes_list,feature_list) :
            if(char==32):
                char=46
            test.write("%s %s\n" %(chr(char), ' '.join(map(str,feature))))

def load_data_from_file(in_file):
    #print in_file
    with open("%s" % in_file) as char_file:
    	ncols = len(char_file.readline().split(' '))
    classes = numpy.loadtxt("%s" %in_file, dtype = str, usecols = [0])
    #print classes
    features = numpy.loadtxt("%s" %in_file, dtype = float, usecols = range(1,ncols))
    #print features
    classes = [ord(x) for x in classes]
    features= numpy.asarray(features, dtype=numpy.float32 )
    classes= numpy.asarray( classes, dtype=numpy.float32 )
    return classes,features