import cv2
import numpy
SEGMENTS_DIRECTION = 0  # vertical axis in numpy


def segments_to_numpy(segments):
    '''given a list of 4-element tuples, transforms it into a numpy array'''
    segments = numpy.array(segments, ndmin=2)
    # each segment in a row
    if SEGMENTS_DIRECTION != 0:
        numpy.transpose(segments)
    return segments


def draw_segments(image, segments, color=(255, 0, 0), line_width=1):
    '''draws segments on image'''
    for segment in segments:
        x, y, w, h = segment
        cv2.rectangle(image, (x, y), (x+w, y+h), color, line_width)


def draw_individual_segment(image, segment, color=(0, 0, 255), line_width=1):
    '''draws a rectangle around the current segment'''
    x, y, w, h = segment
    copy_image = image.copy()
    cv2.rectangle(copy_image, (x, y), (x+w, y+h), color, line_width)
    cv2.imshow('Display', copy_image)


def segment_blocks(segments,inner_segments,euler_list):
    segment_block_list = []
    segment_blocks=[]
    modified_euler_list={}
    central_y={}
    central_x={}
    xmax,ymax,wmax,hmax = numpy.amax(segments,axis=0)
    ordered_segments=numpy.empty([0,4],int)
    start_y=segments[0][1]
    min_y=start_y
    max_y=start_y+segments[0][3]
    bad_flag=0
    valid=0
    for each_segment in segments :
        x,y,w,h = each_segment
        central_y_axis=x+w/2
        central_x_axis=y+h/2
        if(valid==0) :
            min_y=y
            max_y=y+h
        loop_flag=0
        for inner in inner_segments:
            x2,y2,w2,h2=inner
            if([x,y,w,h]==[x2,y2,w2,h2]):
                loop_flag=1
                break
        if loop_flag>0:
            continue
        loop_flag=0
        for segment in segments :
            x1,y1,w1,h1=segment
            if (x==x1 and y==y1 and w==w1 and h==h1):
                continue
            if ((x>=x1) and ((x+w-5)<=(x1+w1)) and y<y1) or (y+h)==y1 :
                if (y+h+5)>=y1 :
                    loop_flag+=1
                    break
        if loop_flag>0:
            continue
        if y>=min_y and y<max_y :
            valid=valid+1
            bad_flag=0
            old_key=str(each_segment.tolist())
            each_segment[1]=min_y
            each_segment[3]=hmax
            new_key=str(each_segment.tolist())
            modified_euler_list[new_key]=euler_list[old_key]
            central_x[new_key]=central_x_axis
            central_y[new_key]=central_y_axis
            segment_blocks.append(each_segment)
        else :
            valid=valid+1
            segment_block_list.append(segment_blocks)
            segment_blocks=[]
            min_y=y
            max_y=y+h
            old_key=str(each_segment.tolist())
            each_segment[1]=min_y
            each_segment[3]=hmax
            new_key=str(each_segment.tolist())
            modified_euler_list[new_key]=euler_list[old_key]
            segment_blocks.append(each_segment)
            central_x[new_key]=central_x_axis
            central_y[new_key]=central_y_axis
            bad_flag+=1
            continue
    if bad_flag==0 :
        segment_block_list.append(segment_blocks)
    for each_block in range(len(segment_block_list)) :
        segment_block_list[each_block]=numpy.array(segment_block_list[each_block], ndmin=2)
        segment_block_list[each_block]=order_segments(segment_block_list[each_block])
        ordered_segments=numpy.concatenate((ordered_segments,segment_block_list[each_block]))
    return ordered_segments,modified_euler_list,central_x,central_y

def order_segments(segments) :
    segments.view('i8,i8,i8,i8').sort(order=['f0'], axis=0)
    return segments
