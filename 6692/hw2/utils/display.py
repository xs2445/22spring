"""
This file contains functions for displaying inferences of the FaceNet MTCNN model.

E6692 Spring 2022s
"""
# import modules
import os
import cv2
import numpy as np
from PIL import Image
import IPython

from .pretrained_deployment import display_image, show_array

DOWNLOADS_PATH = './downloads/'
DATA_PATH = './data/'

def display_images(query, num_images=3):
    """
    Displays the images in the directory './<downloads_path>/<query>'.

    params:
        query (string): image download query
        num_images (int): max numer of images to display

    DO NOT MODIFY THIS FUNCTION
    """
    image_names = os.listdir(os.path.join(DOWNLOADS_PATH, query)) # list image names
    image_count = 1 # initialize image count
    for image_name in image_names: # iterate through image names
        if image_count > num_images: # if image count is larger than num_images, break
            break
        if image_name != '.ipynb_checkpoints': # exclude .ipynb_checkpoints
            image = os.path.join(DOWNLOADS_PATH, query, image_name) # define image path
            display_image(image) # display image with display_image()
            image_count += 1 # increment image count


def display_faces(query, face_detector, num_images=3):
    """
    Display only the faces of the queried images.

    params:
        query (string): image download query
        face_detector (facenet_pytorch.models.mtcnn.MTCNN): face detection model
        num_images (int): max number of images to display

    HINT: your implementation can be similar to display_images() and should use display_image().
    """

    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################
    
    # get the absolute path of images
    image_names = os.listdir(os.path.join(DOWNLOADS_PATH, query))
    # count images as the condition to end the loop
    image_count = 1
    # save faces in the list
    face_list = []
    # for images in the directory
    for image_name in image_names:
        if image_count>num_images:
            break
        # exclude checkpoints
        if image_name != '.ipynb_checkpoints':
            # read image
            image = Image.open(os.path.join(DOWNLOADS_PATH, query, image_name))
            # inference
            img_cropped = face_detector(image)
            # append faces to the list
            if img_cropped is not None:
                face_list.append(img_cropped.numpy())
            print(image_name, 'analyzed!')
            image_count += 1
    
    # diaplay faces
    for faces in face_list:
        # if only 1 face detected, then shape should be (3,w,h)
        if len(faces.shape) == 3:
            # reshape and resize for diaplay
            f = faces.transpose(1,2,0)/255
            display_image(f)
        # if multiple faces detected, then shape should be (n,3,w,h)
        if len(faces.shape) == 4:
            for i in range(faces.shape[0]):
                f = faces[i].transpose(1,2,0)/255
                display_image(f)

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################


def draw_boxes_and_landmarks(frame, boxes, landmarks):
    """
    This function draws bounding boxes and landmarks on a frame. It uses cv2.recangle() to
    draw the bounding boxes and cv2.circle to draw the landmarks.

    See OpenCV docs for more information on cv2.rectangle() and cv2.circle().

    https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
    https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/

    params:
        frame (PIL.Image or np.array): the input frame
        boxes (list): 2D list of bounding box coordinates with shape (num_boxes, 4)
        landmark (list): 3D list of landmark points with shape (num_landmark_groups, 5, 2)

    returns:
        frame (np.array): the frame with bounding boxes and landmarks drawn.
    """
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################
    # color of the rectangle (opencv use GBR)
    color = (0,0,255)
    thickness = 1
    radius = 1
    
    if len(boxes.shape) == 1:
        frame = cv2.rectangle(frame, boxes[0:2], boxes[2:4], color, thickness)
        for j in range(landmarks.shape[0]):
            frame = cv2.circle(frame, landmarks[j], radius, color, thickness)
            
    elif len(boxes.shape) == 2:
        for i in range(boxes.shape[0]):
            start = boxes[i,0:2]
            
            frame = cv2.rectangle(frame, start, boxes[i,2:4], color, thickness)
            for j in range(landmarks.shape[1]):
                frame = cv2.circle(frame, landmarks[i,j], radius, color, thickness)
    
    
    
    
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################

    return frame


def display_detection_and_keypoints(query, face_detector, num_images=3):
    """
    This function displays the bounding boxes and keypoints (landmarks) for the images
    at the query directory. It uses draw_boxes_and_landmarks() to draw the bounding boxes
    and landmarks, then displays the frame.

    params:
        query (string): the query used to download google images.
        face_detector (facenet_pytorch.models.mtcnn.MTCNN): face detection model
        num_images (int): max number of images to display
    """
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    # get the absolute path of images
    image_names = os.listdir(os.path.join(DOWNLOADS_PATH, query))
    # count images as the condition to end the loop
    image_count = 1
    # save faces in the list
    box_list = []
    mark_list = []
    # for images in the directory
    for image_name in image_names:
        if image_count>num_images:
            break
        # exclude checkpoints
        if image_name != '.ipynb_checkpoints':
            # read image
            image = Image.open(os.path.join(DOWNLOADS_PATH, query, image_name))
            # inference
            boxes, _, landmarks = face_detector.detect(image, landmarks=True)
            # append faces to the list
            if boxes is not None:
                box_list.append(boxes)
                mark_list.append(landmarks)
            print(image_name, 'analyzed!')
            image = draw_boxes_and_landmarks(image, boxes, landmarks)
            display_image(image)
            
            image_count += 1
    return box_list, mark_list
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################


def video_inference(video_path, face_detector, max_frames=30):
    """
    This function uses the face detection model to generate a "detected version" of the
    specified video.

    params:
        video_path (string): path to the video to do inference on
        face_detector (facenet_pytorch.models.mtcnn.MTCNN): face detection model
        max_frames (int): the maximum frames to do inference with. Default is 30 frames.

    returns:
        detected_video_path (string): the path to the detected video

    DO NOT MODIFY CODE OUTSIDE OF YOUR IMPLEMENTATION AREA
    """
    video_name = video_path.split('/')[-1].split('.')[0] # get name of video
    detected_video_name = video_name + '-detected.mp4' # append detected name
    detected_video_path = os.path.join(DATA_PATH, detected_video_name) # define detected video path

    v_cap = cv2.VideoCapture(video_path) # initialize the video capture
    fourcc = cv2.VideoWriter_fourcc(*'VP90') # define encoding type
    fps = 30.0 # define frame rate
    video_dims = (960, 540) # define output dimensions
    out = cv2.VideoWriter(detected_video_path, fourcc, fps, video_dims) # initialize video writer

    frame_count = 0 # initialize frame count

    while True:
        frame_count += 1 # increment frame count

        success, frame = v_cap.read() # read frame from video

        if frame_count % 10 == 0:
            print("Frames detected: ", frame_count)

        if not success or frame_count >= max_frames: # break if end of video or max frames is reached
            break

        frame = Image.fromarray(frame) # read frame as PIL Image

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.display.video_inference() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        out.write(np.array(frame)) # write detected frame to output video

    v_cap.release() # release video reader and writer
    out.release()
    cv2.destroyAllWindows()

    return detected_video_path


def webcam_inference(face_detector):
    """
    This function implements the webcam display and performs inference using a face detection
    model.

    param:
        face_detector (facenet_pytorch.models.mtcnn.MTCNN): face detection model

    DO NOT MODIFY CODE OUTSIDE OF YOUR IMPLEMENTATION AREA
    """
    cam = cv2.VideoCapture(0) # define camera stream

    try: # start video feed
        print("Video feed started.")

        while True:
            success, frame = cam.read() # read frame from video stream

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert raw frame from BGR to RGB

            #####################################################################################
            # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
            #####################################################################################

            raise Exception('utils.display.video_inference() not implemented!') # delete me

            #####################################################################################
            # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
            #####################################################################################

            show_array(frame) # display the frame in JupyterLab

            IPython.display.clear_output(wait=True) # clear the previous frame

    except KeyboardInterrupt: # if interrupted
        print("Video feed stopped.")
        cam.release() # release the camera feed
