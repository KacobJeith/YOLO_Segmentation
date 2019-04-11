import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import progressbar
import keyboard

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default=None, required=True, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="FC-DenseNet56", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="MTA_segmentation_labels", required=False, help='The dataset you are using')
args = parser.parse_args()


def initializeNetwork() :

    class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

    num_classes = len(label_values)

    print("\n***** Begin prediction *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
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

    network, _ = model_builder.build_model(args.model, 
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

    if cv2.waitKey(1) & 0xFF == ord('q') or success==False or keyboard.is_pressed('q'):
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

def segmentVideo(sess, network, net_input, label_values):

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
        
        result = predictOnFrame(sess, network, net_input, frame, label_values)
        combined = combineFrameAndLabels(frame, result)
        cv2.imshow('display', combined)

        results_video.write(combined)

    print("Completed Video")
    return


sess, network, net_input, label_values = initializeNetwork()
segmentVideo(sess, network, net_input, label_values)

# image = utils.load_image(args.image)
# result = predictOnFrame(sess, network, net_input, image)