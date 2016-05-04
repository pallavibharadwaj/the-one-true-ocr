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
    print segment
    cv2.rectangle(copy_image, (x, y), (x+w, y+h), color, line_width)
    cv2.imshow('Display', copy_image)


def segment_blocks(segments):
    segment_block_list = []
    segment_blocks=[]
    ordered_segments=numpy.empty([0,4],int)
    start_y=segments[1][1]
    min_y=start_y
    max_y=start_y+segments[1][3]
    bad_flag=0
    print "min_y :",min_y
    print "max_y :", max_y
    for each_segment in segments :
        #print each_segment
        x,y,w,h = each_segment
        if y==min_y or y<=max_y :
            segment_blocks.append(each_segment)
        else :
            min_y=y
            max_y=y+h
            print "min_y :",min_y
            print "max_y :", max_y
            segment_block_list.append(segment_blocks)
            segment_blocks=[]
            segment_blocks.append(each_segment)
            bad_flag+=1
            continue
    if bad_flag==0 :
        segment_block_list.append(segment_blocks)
    for each_block in range(len(segment_block_list)) :
        segment_block_list[each_block]=numpy.array(segment_block_list[each_block], ndmin=2)
        segment_block_list[each_block]=order_segments(segment_block_list[each_block])
        ordered_segments=numpy.concatenate((ordered_segments,segment_block_list[each_block]))
    return ordered_segments

def order_segments(segments) :
    segments.view('i8,i8,i8,i8').sort(order=['f0'], axis=0)
    return segments
