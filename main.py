import cv2
import numpy

SEGMENTS_DIRECTION= 0 # vertical axis in numpy

def segments_to_numpy( segments ):
    '''given a list of 4-element tuples, transforms it into a numpy array'''
    segments= numpy.array( segments, dtype=numpy.uint16, ndmin=2)   #each segment in a row
    segments= segments if SEGMENTS_DIRECTION==0 else numpy.transpose(segments)
    return segments
def draw_segments( image , segments, color=(255,0,0), line_width=1):
        '''draws segments on image'''
        for segment in segments:
            x,y,w,h= segment
            cv2.rectangle(image,(x,y),(x+w,y+h),color,line_width)

image = cv2.imread('alpha.png')
cv2.imshow('Display',image)
copy = image.copy() # Keeping a non-greyscale copy of the image for later use
cv2.waitKey(0)
image = cv2.GaussianBlur(image,(5,5),0) # Blurring the image
cv2.imshow('Display',image)
print("After Guassian blurring")
cv2.waitKey(0)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # Grey scaling the image
cv2.imshow('Display',image) # Displaying the image
print("After Greyscaling")
cv2.waitKey(0)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
'''Thresholding the image (Converts greyscale to binary), using adaptive threshold for best results '''
im2, contours, hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
''' Finding the contours in the image'''
image.fill(255)
cv2.drawContours(image, contours, -1, (0,0,0))
''' Drawing the contours on the image before displaying'''
cv2.imshow('Display',image)
print("After detecting Contours")
cv2.waitKey(0)
segments= segments_to_numpy( [cv2.boundingRect(c) for c in contours] )
''' Convert each segment to x,y,w,h (4-element tuple for numpy)'''
draw_segments(copy,segments)
'''Draw the segments on the copy image (cant add color to greyscaled image)'''
cv2.imshow('Display',copy)
print("After Segmentation")
cv2.waitKey(0)
cv2.destroyAllWindows()