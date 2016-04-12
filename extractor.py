def find_eular(each_segment,segments):
	#Finding the Euler Number
	flag=0
	x1,y1,w1,h1=each_segment
	x2=x1+w1
	y2=y1+h1
	for segment in segments :
		if ((segment==each_segment).all()):
			continue
		x,y,w,h=segment
		if x>x1 and x<x2 and y>y1 and y<y2 and x+w>x1 and x+w<x2 and y+h>y1 and y+h<y2  :
			flag=flag+1
	euler_number=1-flag
	return euler_number

def find_total_pixels(image,segment):
	on_pixel=0
	x,y,w,h=segment
	for i in range(y, y+h) :
		for j in  range(x, x+w) :
			pixel_intensity= image[i,j]	#white(255) or black(0)
			#if the pixel is black, find distance from the axis center
			if pixel_intensity==0 :
				on_pixel+=1
	return on_pixel

def extract_coordinate_based_features(image,on_pixel,segment):
	x,y,w,h=segment
	horizontal_sum=0.0
	vertical_sum=0.0
	horizontal_square_sum=0.0
	vertical_square_sum=0.0
	xy_correlation_sum=0.0
	xxy_sum=0.0
	yyx_sum=0.0
	central_y_axis=x+w/2 # the line is where x = 0
	central_x_axis=y+h/2 # the line is where y = 0
	for i in range(y, y+h) :
		for j in  range(x, x+w) :
			pixel_intensity= image[i,j]	#white(255) or black(0)
			#if the pixel is black, find distance from the axis center
			if pixel_intensity==0 :
				horizontal_dist=j- central_y_axis	#negative for left and positive for right of center
				horizontal_sum+=horizontal_dist	#negative if left heavy
				vertical_dist=central_x_axis- i	#positive for up and negative for down of center
				vertical_sum+=vertical_dist		#positive if up heavy
				horizontal_square_sum+=((horizontal_dist)*(horizontal_dist))
				vertical_square_sum+=((vertical_dist)*(vertical_dist))
				xy_correlation=horizontal_dist*vertical_dist
				xy_correlation_sum+=xy_correlation
				xxy_value=(horizontal_dist*horizontal_dist)*vertical_dist
				yyx_value=(vertical_dist*vertical_dist)*horizontal_dist
				xxy_sum+=xxy_value
				yyx_sum+=yyx_value
	horizontal_mean=horizontal_sum/(w*on_pixel)           #divide by width of the segment
	vertical_mean=vertical_sum/(h*on_pixel)               #divide by height of the segment
	horizontal_variance=horizontal_square_sum/on_pixel
	vertical_variance=vertical_square_sum/on_pixel
	xy_correlation_mean=xy_correlation_sum/on_pixel
	xxy_mean=xxy_sum/on_pixel
	yyx_mean=yyx_sum/on_pixel
	return ([horizontal_mean,x,y,w,h,vertical_mean,horizontal_variance,vertical_variance,xy_correlation_mean,xxy_mean,yyx_mean])


'''def horizontal_edge_features(segment):
	on_pixel=0
	x,y,w,h=segment
	for i in range(y, y+h) :
		for j in  range(x, x+w) :
			pixel_intensity= image[i,j]	#white(255) or black(0)
			#if the pixel is black, find distance from the axis center
			if pixel_intensity==0 :
				if image[i-1][j]==255 :
					horizontal_edge_count+=1
					horizontal_edge_dist=j-central_y_axis
					horizontal_edge_dist_sum+=horizontal_edge_dist
	horizontal_edge_dist_mean = horizontal_edge_dist_sum/horizontal_edge_count
	return horizontal_edge_dist_mean'''