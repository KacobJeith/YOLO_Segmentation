import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import progressbar
import keyboard

from utils import utils, helpers
from builders import model_builder

import sys
from yolo import YOLO, detect_video
from PIL import Image

parser = argparse.ArgumentParser()

################################################################################################################################
########## SEGMENTATION ARGUMENTS 
################################################################################################################################

parser.add_argument('--video', type=str, default=None, required=False, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model_segmentation', type=str, default="FC-DenseNet56", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="MTA_segmentation_labels", required=False, help='The dataset you are using')
parser.add_argument("--skip_frames", type=int, default=9, help="name of file containing labels")

################################################################################################################################
########## YOLO ARGUMENTS 
################################################################################################################################

parser.add_argument('--model_yolo', type=str, help='path to model weight file, default ' + YOLO.get_defaults("model_path"))
parser.add_argument('--anchors', type=str, help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path"))
parser.add_argument('--classes', type=str, help='path to class definitions, default ' + YOLO.get_defaults("classes_path"))
parser.add_argument('--gpu_num', type=int, default=1, help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num")))
parser.add_argument('--image', default=False, action="store_true",help='Image detection mode, will ignore all positional arguments')
parser.add_argument("--input", nargs='?', type=str,required=False,default='./path2your_video',help = "Video input path")
parser.add_argument("--output", nargs='?', type=str, default="", help = "[Optional] Video output path")

args = parser.parse_args()

################################################################################################################################
########## SEGMENTATION FUNCTIONS 
################################################################################################################################


def initializeNetwork() :

    class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

    num_classes = len(label_values)

    print("\n***** Begin prediction *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model_segmentation)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Classes -->", num_classes)
    print("Video -->", args.video)

    # Initializing network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

    network, _ = model_builder.build_model(args.model_segmentation, 
                                            net_input=net_input,
                                            num_classes=num_classes,
                                            crop_width=args.crop_width,
                                            crop_height=args.crop_height,
                                            is_training=False)

    sess.run(tf.global_variables_initializer())

    print('Loading model checkpoint weights')
    saver=tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, args.checkpoint_path)

    return sess, network, net_input, label_values

def predictOnFrame(sess, network, net_input, image, label_values):
    
    resized_image =cv2.resize(image, (args.crop_width, args.crop_height))
    input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})
    run_time = time.time()-st

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    final = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)

    return final

def writeImage(result, file_name) :

    file_name = utils.filepath_to_name(args.image)
    cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

    print("")
    print("Finished!")
    print("Wrote image " + "%s_pred.png"%(file_name))

def combineFrameAndLabels(frame,labels) :
    colored_labels = cv2.applyColorMap(labels, cv2.COLORMAP_JET)
    resized_overlay= cv2.resize(colored_labels, frame.shape[0:2][::-1]) 
    combined = cv2.addWeighted(frame,0.8,resized_overlay,1.0,0)

    return combined

def startProgressBar(length):
    bar = progressbar.ProgressBar(maxval=length, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    return bar

def initializeVideo(cap) :

    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    filename = os.path.basename(args.video).split('.')[0]
    
    results_video = cv2.VideoWriter(filename + '_both.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (int(width * 2),int(height)))
    return results_video, length

def checkForInterruption(success, cap, bar, results_video) :

    if cv2.waitKey(1) & 0xFF == ord('q') or success==False or keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
        cap.release()
        results_video.release()
        cv2.destroyAllWindows()
        bar.finish()
        return True
    
    return False

def incrementProgress(frameIndex, bar, skip):

    frameIndex += skip + 1
    
    try:
        bar.update(frameIndex)
        return frameIndex
    except:
        bar.finish()
        return frameIndex


################################################################################################################################
########## YOLO FUNCTIONS 
################################################################################################################################

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def instantiate():
    FLAGS = args

    if FLAGS.image:
        
        print("Image detection mode")

        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)

        detect_img(YOLO(**vars(FLAGS)))

    elif "input" in FLAGS:
        
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    
    else:
        
        print("Must specify at least video_input_path.  See usage with --help.")

################################################################################################################################
########## YOLOSEG FUNCTIONS 
################################################################################################################################

def displayFourResults(wait, im1, im2, im3, im4):
    
    numpy_horizontal = np.hstack((im1, im2))
    numpy_horizontal_concat = np.concatenate((im1, im2), axis=1)
    
    numpy_horizontal_row2 = np.hstack((im3, im4))
    numpy_horizontal_concat_row2 = np.concatenate((im3, im4), axis=1)

    numpy_giant = np.vstack((numpy_horizontal_concat, numpy_horizontal_concat_row2))
    numpy_giant_concat = np.concatenate((numpy_horizontal_concat, numpy_horizontal_concat_row2), axis=2)

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('display', 1600, 1000)
    cv2.imshow('display', numpy_giant_concat)

    if wait :
        cv2.waitKey()

def displayTwoResults(wait, im1, im2, name='double_display'):
    
    numpy_horizontal = np.hstack((im1, im2))
    numpy_horizontal_concat = np.concatenate((im1, im2), axis=1)
    
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1600, 500)
    cv2.imshow(name, numpy_horizontal_concat)

    if wait :
        cv2.waitKey()
    
    return numpy_horizontal_concat


def drawLargestContour(segmented) :
    
    binary = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx_area = 0
    max_index = 0
    for index, cont in enumerate(contours):
        area = cv2.contourArea(cont) 
        if area > mx_area and area > 1:
            mx_area = area
            max_index = index

    cv2.drawContours(segmented, contours, max_index, (0,0,255))

    height, width = segmented.shape[:2]
    primary_binary = np.zeros((height,width,3), np.uint8)
    
    cv2.drawContours(primary_binary, contours, max_index, (255,255,255), -1)

    primary_contour = contours[max_index]

    return segmented, primary_binary, primary_contour

def drawOverlappingOnBinary(binary, buslane, boxes, lane_roi_contour) :
    overlapping = binary.copy()

    for box in boxes :
        top, left, bottom, right = box

        thisObj_circ = binary.copy()
        thisObj_rect = binary.copy()
        cv2.rectangle(thisObj_rect,(left,top),(right,bottom),(255,255,255),3)
        center_point = (betweenTheseTwo(left, right),betweenTheseTwo(top, bottom))
        radius_hor = int(0.7 * (right - left)/2 )
        radius_ver = int(0.7 * (bottom - top)/2 )
        
        #hor circle
        # cv2.circle(thisObj_circ, center_point , radius_hor, (255,255,255), -1)
        # cv2.circle(thisObj_circ, center_point , radius_ver, (255,255,255), -1)
        # cv2.EllipseBox(thisObj_circ, box, color, thickness=1, lineType=8, shift=0)
        cv2.rectangle(thisObj_circ,(left,top),(right,bottom),(210,210,210),-1)

        thisObj = thisObj_circ #& thisObj_rect

        if checkForOverlap(buslane, thisObj) :
            if checkPointsInROI(lane_roi_contour, box) > 1:
                cv2.rectangle(overlapping,(left,top),(right,bottom),(210,210,210),-1)
            # overlapping += thisObj
            # cv2.circle(overlapping, (betweenTheseTwo(left, right),betweenTheseTwo(top, bottom)), int((right - left)/2), (210,210,210), -1)

    return overlapping

def checkPointsInROI(lane_roi_contour, box) :
    top, left, bottom, right = box
    included = 0
    points = [(top,left), (top,right), (bottom, left), (bottom, right)]

    for point in points :
        if cv2.pointPolygonTest(lane_roi_contour, point, False) >= 0:
            included += 1
    
    print('points included: ', included)

    return included


def betweenTheseTwo(val1, val2) :
    return int(val1 + (val2 - val1)/2)

def checkForOverlap(mask1, mask2) :
    ret,bw1 = cv2.threshold(mask1,127,255,cv2.THRESH_BINARY)
    ret,bw2 = cv2.threshold(mask2,127,255,cv2.THRESH_BINARY)
    test = bw1 & bw2
    gray_image = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    nnz = cv2.countNonZero(gray_image)

    return nnz > 0

def displaySingle(image, name='single_display') :

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1500, 800)
    cv2.imshow(name, image)

def extrapolateLaneAndDraw(image, contour) :
    #### CONVEX HULL ####

    area = cv2.contourArea(contour) 
    imageWithTriangle = image.copy()

    if area > 2000 :

        hull = cv2.convexHull(contour)
        cv2.drawContours(image, [hull], -1, (255, 255, 255), -1)

        lane_target = getLaneTargetPoint(image, contour)

        ## Extrapolate forward a bit ##
        left, right, top, bottom = getExtremePoints(contour)
        pts = np.array( [left, right, lane_target] )
        imageWithTriangle = cv2.fillPoly(imageWithTriangle,[pts],(255,255,255))
    
    return imageWithTriangle, image

def drawLaneROIMask(binary) :

    
    height, width = binary.shape[:2]

    bottomMargin = round(height - height/2) + 200

    bottomLeft = (0, height)
    bottomUpper = (0, bottomMargin)

    topMargin = 100
    topLeft = (round(width/2 - topMargin), 0)
    topRight = (round(width/2 + topMargin), 0)
    bottomRightUpper = (width,bottomMargin)
    bottomRight = (width, height)

    pts = np.array( [bottomLeft, bottomUpper, topLeft,topRight, bottomRightUpper, bottomRight ] )
    
    binaryROI = cv2.fillPoly(binary,[pts],(255,255,255))

    return binaryROI

def drawCircleAtPoint(image, point):
    return cv2.circle(image,point, 20, (0,0,255), -1)


def binaryOfThisSize(image) :
    height, width = image.shape[:2]
    black_im = np.zeros((height,width,3), np.uint8)
    
    ret,binary = cv2.threshold(black_im,127,255,cv2.THRESH_BINARY)

    return binary

def getCenterOfMass(contour) :
    
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return cx, cy

def getLaneTargetPoint(frame, contour):
    cx, cy = getCenterOfMass(contour)

    height, width, channels = frame.shape

    frame_center_x = width/2
    target_x = betweenTheseTwo(frame_center_x, frame_center_x + (cx - frame_center_x)/2)
    # target_y = betweenTheseTwo(betweenTheseTwo(height/6, cy) , height/6)
    left, right, top, bottom = getExtremePoints(contour)
    target_y = height/8 #betweenTheseTwo(betweenTheseTwo(top[1], height/7), height/7)
    
    lane_target = (round(target_x), round(target_y))

    return lane_target
    
def getExtremePoints(c) :
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])

    return left, right, top, bottom

def processVideo(yolo_instance, sess, network, net_input, label_values):

    cap = cv2.VideoCapture(args.video)
    results_video, length = initializeVideo(cap)
    bar = startProgressBar(length)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    success, frame = cap.read()

    frameIndex = 0


    laneroimask = binaryOfThisSize(frame)
    laneroimask = drawLaneROIMask(laneroimask)
    # _ , binary = cv2.threshold(laneroimask,127,255,cv2.THRESH_BINARY)
    binary = cv2.cvtColor(laneroimask, cv2.COLOR_BGR2GRAY)
    lane_roi_contour, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    lane_roi_contour = lane_roi_contour[0]

    while success : 

        if checkForInterruption(success, cap, bar, results_video) : 
            success = False
            break
        
        segmentation_result = predictOnFrame(sess, network, net_input, frame, label_values)
        segmentation_result_resized = cv2.resize(segmentation_result, frame.shape[0:2][::-1]) 

        resized_seg, resized_lane_binary, primary_contour = drawLargestContour(segmentation_result_resized)

        # resized_lane_binary = cv2.resize(primary_binary, frame.shape[0:2][::-1]) 
        # resized_seg = cv2.resize(seg_contour, frame.shape[0:2][::-1]) 

        boxes_binary = binaryOfThisSize(frame)

        # segmentation_overlay = combineFrameAndLabels(frame, seg_contour)

        # frame_seg_size = cv2.resize(frame, segmentation_result.shape[0:2][::-1]) 
        lane_binary = binaryOfThisSize(resized_seg)
        lane_prediction, only_original_prediction = extrapolateLaneAndDraw(lane_binary, primary_contour)
        # lane_prediction = cv2.resize(lane_prediction, frame.shape[0:2][::-1]) 
        lane_overlay = combineFrameAndLabels(frame, only_original_prediction)

        # displaySingle(lane_overlay)

        yolo_result, out_boxes = yolo_instance.detect_image(Image.fromarray(frame), True)

        binary_boxes = drawOverlappingOnBinary(boxes_binary, lane_prediction, out_boxes, lane_roi_contour)

        # everything = combineFrameAndLabels(np.array(yolo_result) , segmentation_overlay)
        everything_boxes = combineFrameAndLabels(np.array(yolo_result) , binary_boxes )

        # displaySingle(everything_boxes)

        # cv2.imshow('everything', everything_boxes)
        combined = displayTwoResults(False, lane_overlay, everything_boxes)

        results_video.write(combined)

        frameIndex = incrementProgress(frameIndex, bar, args.skip_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        success,frame = cap.read()

    print("Completed Video")
    return

if __name__ == '__main__':
    sess, network, net_input, label_values = initializeNetwork()
    yolo_instance = YOLO(**vars(args))

    processVideo(yolo_instance, sess, network, net_input, label_values)
    # segmentVideo(sess, network, net_input, label_values)
    print('Closing GPU Sessions...')

    sess.close()
    yolo_instance.close_session()

    print('Mission Accomplished')




# image = utils.load_image(args.image)
# result = predictOnFrame(sess, network, net_input, image)