import argparse
import os, sys
from utils.yoloTRT import TrtYOLO
import numpy as np
import cv2
import time
from utils.utils import draw_bboxes

VIDEO_PATH = "test-lowres.mp4"
SAVING_PATH = "test-lowres-tensorrt-detected.avi"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
INPUT_FRAME_WIDTH = 640
INPUT_FRAME_HEIGHT = 480

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--trt_file', type=str, required=True,
        help=('Specify TRT engine file path'))
    args = parser.parse_args()
    
    # load trt model
    trtyolo_model = TrtYOLO(args.trt_file)
    
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
        # start record time
        t_start = time.time()
        # inferencing
        boxes, scores, classes = trtyolo_model.detect(frame)
        t_mean += time.time() - t_start
        # draw boxes on img
        frame = draw_bboxes(frame, boxes, scores)
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