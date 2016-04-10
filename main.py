import cv2
import numpy

SEGMENTS_DIRECTION= 0 # vertical axis in numpy

def segments_to_numpy( segments ):
    '''given a list of 4-element tuples, transforms it into a numpy array'''
    segments= numpy.array( segments, dtype=numpy.uint16 , ndmin=2)   #each segment in a row
    segments= segments if SEGMENTS_DIRECTION==0 else numpy.transpose(segments)
    return segments

def draw_segments( image , segments,color=(255,0,0), line_width=1):
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
cv2.drawContours(image, contours, -1, (0,0,255))
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
#Is removing unnecessary segments from the segment list required?
count=0 #counting number of segments within a given segment
#Finding the outer bounding box to find segments within a segment 
xmin,ymin,wmin,hmin= numpy.amin(segments, axis=0) 
xmax,ymax,wmax,hmax= numpy.amax(segments, axis=0) 
print "number of segments:{}".format(len(segments)) #number of segments for reference and confirmation
print "Feature Data: Euler Number, Horizontal Mean, Vertical Mean"
count=0
feature_list =[]
for each_segment in segments :
	#Finding the Euler Number
	flag=0
	x1,y1,w1,h1=each_segment
	if (each_segment==[xmin,ymin,wmax,hmax]).all():
			continue
	x2=x1+w1
	y2=y1+h1
	for segment in segments :
		if ((segment==each_segment).all()):
			continue
		x,y,w,h=segment
		if x>x1 and x<x2 and y>y1 and y<y2 and x+w>x1 and x+w<x2 and y+h>y1 and y+h<y2  :
			#print "okay"
			flag=flag+1
			count=count+1
	euler_number=1-flag

	#horizontal mean list and vertical mean list
	horizontal_sum=0
	vertical_sum=0
	x,y,w,h=each_segment
	#center of the segment
	segment_centres=x+w/2,y+h/2
	central_y_axis=x+w/2 # the line is where x = 0
	central_x_axis=y+h/2 # the line is where y = 0
	for i in range(x, x+w) :
		for j in  range(y, y+h) :
			pixel_intensity= image[j,i]	#white(255) or black(0)
			#print pixel_intensity
			#if the pixel is black, find distance from the segment center
			if pixel_intensity==0 :
				horizontal_dist=i- central_y_axis	#negative for left and positive for right of center
				horizontal_sum+=horizontal_dist	#negative if left heavy
				vertical_dist=central_x_axis- j	#positive for up and negative for down of center
				vertical_sum+=vertical_dist		#positive if up heavy
	horizontal_mean=horizontal_sum/w           #divide by width of the segment
	vertical_mean=vertical_sum/h               #divide by height of the segment
	# Converting all the feature data into a tuple
	feature_list.append([euler_number,horizontal_mean,vertical_mean])
print feature_list
cv2.waitKey(0)
cv2.destroyAllWindows()