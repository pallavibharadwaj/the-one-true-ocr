The One True OCR
=================

##Requirements##
  * OpenCV 3.1.0
  * Python-dev
  * Pip
  * Numpy
  * SciPy
  * Scikit-Learn

#### Requirements Installation ####
  * OpenCV 3.1.0 :- http://opencv.org/downloads.html
  * Rest of the requirements : run ```make install``` in the root directory

### Usage Details ###

#### To run ocr on an image ####

```
python toto.py --model <MODEL> <image_to_test>
```

#### To generate ground data for an image  ####
```
from files import generate_ground_data

generate_ground_data(<path_to_image>)
```

#### To train the model ####

```
from ocr import train

train(<list_of_path_to_images>)
```


#### To Classify/Predict the data ####
```
from ocr import knnModel, SVMModel

# For KNN classification

knnModel(<training_features>,<training_classes>,<test_features>)

# For SVM Prediction

SVMModel(<training_features>,<training_classes>,<test_features>)
```
