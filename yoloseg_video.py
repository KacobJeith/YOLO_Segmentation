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
    
    results_video = cv2.VideoWriter(filename + '_buslane.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (int(width),int(height)))
    return results_video, length

def checkForInterruption(success, cap, bar, results_video) :

    if cv2.waitKey(1) & 0xFF == ord('q') or success==False or keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
        cap.release()
        results_video.release()
        cv2.destroyAllWindows()
        bar.finish()
        return True
    
    return False

def incrementProgress(frameIndex, bar):

    frameIndex += 1
    bar.update(frameIndex)

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

def displayResults(wait, im1, im2):
    
    numpy_horizontal = np.hstack((im1, im2))
    numpy_horizontal_concat = np.concatenate((im1, im2), axis=1)
    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('display', 1600, 900)
    cv2.imshow('display', numpy_horizontal_concat)

    if wait :
        cv2.waitKey()

def processVideo(yolo_instance, sess, network, net_input, label_values):

    cap = cv2.VideoCapture(args.video)
    results_video, length = initializeVideo(cap)
    bar = startProgressBar(length)
    success, frame = cap.read()

    frameIndex = 0

    while success : 
        frameIndex = incrementProgress(frameIndex, bar)
        success, frame = cap.read()

        if checkForInterruption(success, cap, bar, results_video) : 
            success = False
            break
        
        segmentation_result = predictOnFrame(sess, network, net_input, frame, label_values)
        # segmentation_overlay = combineFrameAndLabels(frame, segmentation_result)

        yolo_result = yolo_instance.detect_image(Image.fromarray(frame))

        everything = combineFrameAndLabels(np.array(yolo_result) , segmentation_result)
        
        displayResults(False, frame, everything)

        # results_video.write(yolo_result)

    print("Completed Video")
    return

if __name__ == '__main__':
    sess, network, net_input, label_values = initializeNetwork()
    yolo_instance = YOLO(**vars(args))

    processVideo(yolo_instance, sess, network, net_input, label_values)
    # segmentVideo(sess, network, net_input, label_values)
    print('SUCCESSFULLY COPIED FUNCTIONS TOGETHER')

# image = utils.load_image(args.image)
# result = predictOnFrame(sess, network, net_input, image)