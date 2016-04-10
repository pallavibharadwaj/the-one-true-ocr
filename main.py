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

#Is removing unnecessary segments from the segment list required?

count=0 #counting number of segments within a given segment
#Finding the outer bounding box to find segments within a segment 
xmin,ymin,wmin,hmin= numpy.amin(segments, axis=0) 
xmax,ymax,wmax,hmax= numpy.amax(segments, axis=0) 
euler_number_list=[]

print "number of segments:{}".format(len(segments)) #number of segments for reference and confirmation

count=0
for each_segment in segments :
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
	euler_number_list.append(1-flag)
print count

print "EULER LIST"
print euler_number_list

#x,y,w,h already there
#will push the number of on pixels by 12 maybe
#Will push all the features into a numpy array later

#6th and 7th) horizontal mean list and vertical mean list

#finding number of rows and columns in the image to deal with out-of-bounds
no_of_rows,no_of_cols=image.shape
print "No of rows in the image: {}".format(no_of_rows)
print "No of columns in the image: {}".format(no_of_cols)

horizontal_mean_list=[]
vertical_mean_list=[]
for segment in segments :
	horizontal_mean=0
	vertical_mean=0
	x,y,w,h=segment

	#center of the segment
	segment_centers=(x+(x+w))/2,(y+(y+h))/2
	center_x=(x+(x+w))/2
	center_y=(y+(y+h))/2
	#print "{} {}".format(center_x,center_y)

	for i in range(x, x+w) :
		for j in  range(y, y+h) :
			#verifying if the position is within the range
			if i<no_of_rows and j<no_of_cols :	
				pixel_intensity= image[i,j]	#white(255) or black(0)
				#print pixel_intensity
				#if the pixel is black, find distance from the segment center
				if pixel_intensity==0 :
					horizontal_dist=i-center_x	#negative for left and positive for right of center
					horizontal_mean+=horizontal_dist	#negative if left heavy
					vertical_dist=j-center_y	#positive for up and negative for down of center
					vertical_mean+=vertical_dist		#positive if up heavy
	horizontal_mean=horizontal_mean/w           #divide by width of the segment
	vertical_mean=vertical_mean/h               #divide by height of the segment
	horizontal_mean_list.append(horizontal_mean)	
	vertical_mean_list.append(vertical_mean)
		
print "HORIZONTAL MEAN LIST"
print len(horizontal_mean_list)
print horizontal_mean_list
print "VERTICAL MEAN LIST"
print len(vertical_mean_list)
print vertical_mean_list

draw_segments(copy,segments)
'''Draw the segments on the copy image (cant add color to greyscaled image)'''
cv2.imshow('Display',copy)
print("After Segmentation")
cv2.waitKey(0)
cv2.destroyAllWindows()