import argparse
import os, sys
import numpy as np
import cv2
import time
import torch
from darknet_utils.darknet_to_pytorch import load_darknet_as_pytorch
from darknet_utils.torch_utils import detect
from darknet_utils.utils import plot_boxes_cv2

VIDEO_PATH = "test-lowres.mp4"
SAVING_PATH = "test-lowres-pytorch-detected.avi"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
INPUT_FRAME_WIDTH = 640
INPUT_FRAME_HEIGHT = 480

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', '--weight_file', type=str, required=True,
        help=('Specify weight file path'))
    parser.add_argument(
        '-c', '--configuration_file', type=str, required=True,
        help=('Specify configuration file path'))
    args = parser.parse_args()
    
    # load trt model
    pytorch_model = load_darknet_as_pytorch(
        args.configuration_file, 
        args.weight_file)
    pytorch_model.cuda()
    
    count = 0
    t_mean = 0
    t_mean_e2e = 0
    
    # load testing video
    cap = cv2.VideoCapture(VIDEO_PATH)
    # get the fps of original video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # write video
    out = cv2.VideoWriter(SAVING_PATH,
                          cv2.VideoWriter_fourcc('M','J','P','G'), 
                          fps, (FRAME_WIDTH,FRAME_HEIGHT))
    t_start_e2e = time.time()
    
    # start inferencing
    print("Inferencing...")
    while cap.isOpened():
        # read one frame of video
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Exiting ...")
            break
        # count frames
        count += 1
        # resize the input frame to smaller size
        frame = cv2.resize(frame,(FRAME_WIDTH,FRAME_HEIGHT))
        img = cv2.resize(frame,(INPUT_FRAME_WIDTH,INPUT_FRAME_HEIGHT))
        # start record time
        t_start = time.time()
        # inferencing
        boxes = detect(pytorch_model, img, 0.5, 0.5)
#         print(boxes)
        t_mean += time.time() - t_start
        # draw boxes on img
        frame = plot_boxes_cv2(frame, boxes[0])
        # write results to video
        out.write(frame)
        if count % 100 == 99:
            print("Processed {} frames".format(count+1))
    cap.release()
    out.release()

    t_mean_e2e = time.time() - t_start_e2e
    
    # average inferencing time for one image (fps)
    t_mean /= count
    # end-to-end fps
    t_mean_e2e /= count
    
    print("Inference Speed (fps):{:4f}, End-to-End Speed (fps):{:4f}".format(1/t_mean, 1/t_mean_e2e))
    

if __name__ == "__main__":
    main()
#     main2()