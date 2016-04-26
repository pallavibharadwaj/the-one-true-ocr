import numpy
import glob
from extractor import get_feature_list, get_class_list
for txt_file in glob.iglob('data/*.txt') :
    fp = txt_file
#print fp
def generate_ground_data(image,copy, segments):
    feature_list= get_feature_list(image, segments)
    classes_list= get_class_list(copy, segments)
    with open("%s" % fp,'wb') as test :
        for char,feature in zip(classes_list,feature_list) :
            if(char==32):
                char=46
            test.write("%s %s\n" %(chr(char), ' '.join(map(str,feature))))

def load_data_from_file():
    with open("%s" % fp) as char_file:
    	ncols = len(char_file.readline().split(' '))
    classes = numpy.loadtxt("data/alpha.txt", dtype = str, usecols = [0])
    features = numpy.loadtxt("%s" % fp, dtype = float, usecols = range(1,ncols))
    classes = [ord(x) for x in classes]
    features= numpy.asarray(features, dtype=numpy.float32 )
    classes= numpy.asarray( classes, dtype=numpy.float32 )
    return classes, features