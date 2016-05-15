import cv2
from segmentation import draw_individual_segment
import numpy
from segmentation import segments_to_numpy


def find_euler_and_inner_segments(segments,return_flag):
    '''Finding the Euler Numbers and inner_segments for the given image or set of segments'''
    euler_list = {}
    inner_segments = []
    for each_segment in segments:
        flag = 0
        counter=0
        x1, y1, w1, h1 = each_segment
        x2 = x1+w1
        y2 = y1+h1
        for segment in segments:

            if ((segment == each_segment).all()):
                continue
            x, y, w, h = segment
            if (
                x >= x1 and
                x <= x2 and
                y >= y1 and
                y <= y2 and
                x+w >= x1 and
                x+w <= x2 and
                y+h >= y1 and
                y+h <= y2
            ) :
                flag = flag+1
                if flag != 0:
                    inner_segments.append(segment)
            counter=counter+1
        euler_number = 1-flag
        seg_key = str(each_segment.tolist())
        euler_list[seg_key] = euler_number
    if return_flag==1 :
        return euler_list, inner_segments
    else :
        return inner_segments

def find_total_on_pixels(image, segment):
    '''Finding the total number of on_pixels in a segment'''
    on_pixel = 0
    x, y, w, h = segment
    for i in range(y, y+h):
        for j in range(x, x+w):
            pixel_intensity = image[i, j]  # white(255) or black(0)
            # if the pixel is black, find distance from the axis center
            if pixel_intensity == 0:
                on_pixel += 1
    return on_pixel

central_y=[]
central_x=[]

def extract_coordinate_based_features(image, on_pixel, segment ,org_x):
    '''Finding all the features based of the x,y of on_pixels in a segment'''
    x, y, w, h = segment
    horizontal_sum = 0.0
    vertical_sum = 0.0
    horizontal_square_sum = 0.0
    vertical_square_sum = 0.0
    xy_correlation_sum = 0.0
    xxy_sum = 0.0
    yyx_sum = 0.0
    central_y_axis = x+w/2  # the line is where x = 0
    central_x_axis=org_x[str(segment.tolist())]
    central_y.append(central_y_axis)
    central_x.append(central_x_axis)
    for i in range(y, y+h):
        for j in range(x, x+w):
            pixel_intensity = image[i, j]    # white(255) or black(0)
            # if the pixel is black, find distance from the axis center
            if pixel_intensity == 0:
                horizontal_dist = j - central_y_axis
                horizontal_sum += horizontal_dist
                vertical_dist = central_x_axis - i
                vertical_sum += vertical_dist
                horizontal_square_sum += ((horizontal_dist)*(horizontal_dist))
                vertical_square_sum += ((vertical_dist)*(vertical_dist))
                xy_correlation = horizontal_dist*vertical_dist
                xy_correlation_sum += xy_correlation
                xxy_value = (horizontal_dist*horizontal_dist)*vertical_dist
                yyx_value = (vertical_dist*vertical_dist)*horizontal_dist
                xxy_sum += xxy_value
                yyx_sum += yyx_value
    horizontal_mean = horizontal_sum/(w*on_pixel)
    vertical_mean = vertical_sum/(h*on_pixel)
    horizontal_variance = horizontal_square_sum/on_pixel
    vertical_variance = vertical_square_sum/on_pixel
    xy_correlation_mean = xy_correlation_sum/on_pixel
    xxy_mean = xxy_sum/on_pixel
    yyx_mean = yyx_sum/on_pixel
    edge_list = feature_extraction_edges(image,segment,on_pixel)
    final_return_data = [
        horizontal_mean, vertical_mean, horizontal_variance,
        vertical_variance, xy_correlation_mean, xxy_mean, yyx_mean]
    final_return_data = final_return_data+edge_list
    return ( final_return_data)


def get_char(image, segment):
    '''getting the character from user for the segment'''
    key_list = []
    draw_individual_segment(image, segment)
    key = cv2.waitKey(0)
    key%=256
    while key<=32 or key>=127 : #invalid characters are pressed
        if key==255 : #shift key
            key=cv2.waitKey(0) #wait for another character(special symbols)
            key%=256
        else :
            key = cv2.waitKey(0)
            key%=256
    return chr(key)


def get_feature_list(image ,segments,euler_list,org_x,org_y):
    feature_list = []
    classes_list = []
    spaces_list = []#
    before_x=[]
    before_y=[]
    inner_segments=find_euler_and_inner_segments(segments,0)
    #add code to remove inner segments
    segment_count = 0
    for each_segment in segments:
        bad_flag=0
        x,y,w,h=each_segment
        for inner in inner_segments:
            x1,y1,w1,h1=inner
            if ([x,y,w,h]==[x1,y1,w1,h1]):
                bad_flag=1
                continue
        if(bad_flag==1):
            continue
        seg_key = str(each_segment.tolist())
        euler_number = euler_list[seg_key]
        segment_count+=1
        on_pixel = find_total_on_pixels(image, each_segment)
        if on_pixel == 0 :
            continue     #empty segment
        coordinate_features = extract_coordinate_based_features(
            image,
            on_pixel,
            each_segment,org_x)
        # Converting all the feature data into a tuple
        segment_features = [euler_number,on_pixel]+coordinate_features
        final_data = segment_features
        feature_list.append(final_data)
        before_x.append(org_x[str(each_segment.tolist())])
        before_y.append(org_y[str(each_segment.tolist())])

    for i in range(len(segments)-1):
        temp=segments[i+1][0]-(segments[i][0]+segments[i][2])
        spaces_list.append(temp)
    return feature_list,spaces_list


def get_class_list(image,segments):
    classes_list = []
    inner_segments=find_euler_and_inner_segments(segments,0)
    #add code to remove inner segments
    segment_count = 0
    for each_segment in segments:
        bad_flag=0
        x,y,w,h=each_segment
        for inner in inner_segments:
            x1,y1,w1,h1=inner
            if ([x,y,w,h]==[x1,y1,w1,h1]):
                bad_flag=1
                continue
        if(bad_flag==1):
            continue
        char_data = get_char(image, each_segment)
        classes_list.append(ord(char_data))
    return classes_list

def feature_extraction_edges(image,segment,on_pixel):
    counter=0.0
    counter1=0.0
    edge_list = []
    x,y,w,h=segment
    horizontal_edge_dist_sum=0.0
    vertical_edge_dist_sum=0.0
    central_y_axis = x+w/2  # the line is where x = 0
    central_x_axis = y+h/2
    for i in range(y,y+h):
        for j in range(x,x+w):
            pixel_old=image[i,j]
            next_horizontal_pixel=image[i,j+1]
            if (pixel_old == 255) and (next_horizontal_pixel== 0) :
                    counter+=1
                    horizontal_edge_dist=j-central_y_axis
                    horizontal_edge_dist_sum+=horizontal_edge_dist
    for j in range(x,x+w):
        for i in range(y,y+h):
            pixel_old_2=image[i,j]
            next_vertical_pixel=image[i+1,j]
            if (pixel_old_2 == 0) and (next_vertical_pixel== 255) :
                    counter1+=1
                    vertical_edge_dist=i-central_x_axis
                    vertical_edge_dist_sum+=vertical_edge_dist
    horizontal_mean_edge = counter/on_pixel
    vertical_mean_edge = counter1/on_pixel
    horizontal_edge_dist_mean = horizontal_edge_dist_sum/counter
    vertical_edge_dist_mean = vertical_edge_dist_sum/counter1
    return ([horizontal_mean_edge,vertical_mean_edge,vertical_edge_dist_mean, horizontal_edge_dist_mean])
