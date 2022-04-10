"""
This file contains the functions for generating Darknet datasets. 

E6692 Spring 2022
"""
import os
import cv2
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

from .utils import del_folder_contents, parse_config, load_test_config, read_validation_video_names
from .load_annotations import load_annotation_objects, get_objects_in_frame, get_frame_bboxes

from darknet_utils.utils import plot_boxes_cv2

FRAME_STEP = 6

def make_darknet_dataset(cfg_path, class_groups=None):
    """
    This function generates a dataset in the format specified by the
    obj.data file

    cfg_path: path to dataset configuration file
    class_groups: dictionary of classes to be grouped together
    
        Example: if we want to group all vehicle classes into the class 'vehicle'
        
        class_groups = {'vehicle' : ['car', 'bus', 'truck']}
    """
    
    print("Decoding evaluation parameters. (Video validation)")
    # decode config file
    options = parse_config(cfg_path, cfg_type='darknet_dataset')
    
    obj_data_filename = str(options["obj_data_filename"])
    train_path = str(options["train_path"])
    val_path = str(options["val_path"])
    
    videos_path = str(options["videos_path"])
    labels_path = str(options["labels_path"])
    
    val_video_names_path = str(options["val_video_names"])
    
    if options["max_vids"] != 'None':
        max_vids = int(options["max_vids"])
    else:
        max_vids = None
        
    if options["max_frames"] != 'None':
        max_frames = int(options["max_frames"])
    else:
        max_frames = None
        
    with open(obj_data_filename, 'r') as obj_data: # get configurations from .data file
        lines = obj_data.readlines()
        lines = [line for line in lines if line[0] not in ["#", "\n"]]
        num_classes = int(lines[0].split('=')[1].strip())
        train_txt_paths = lines[1].split('=')[1].strip()
        val_txt_paths = lines[2].split('=')[1].strip()
        class_names_path = lines[3].split('=')[1].strip()
    
    # if os.getcwd().split('/')[-1] != 'darknet':
    #     class_names_path = class_names_path[1:]
    #     train_txt_paths = train_txt_paths[1:]
    #     val_txt_paths = val_txt_paths[1:]
    
    with open(class_names_path, 'r') as classes: # get class dictionary
        class_dict = {}
        for index, line in enumerate(classes.readlines()):
            class_dict.update({ line.strip() : str(index) })
            
    # print("class dict:",class_dict)
    # for k, v in class_dict.items():
    #     print("***",k,"***",v,"***")
            
    if class_groups is not None:
        # get index of 'superclass' in class_dict (line number in .names file) for each entry to class_groups
        for superclass in list(class_groups.keys()):
            original_classes = class_groups[superclass]
            for original_class in original_classes:
                # print(original_class)
                class_dict.update({ original_class : class_dict[superclass] })
                            
            
    if not os.path.exists(train_path): # create train and val paths if they don't exist. Empty if they do exist
        os.mkdir(train_path)
    else: 
        del_folder_contents(train_path)
    
    if not os.path.exists(val_path): 
        os.mkdir(val_path)
    else: 
        del_folder_contents(val_path)
        
    vids = 0
    frames = 0 # global frame count (accross multiple videos)
    
    train_image_list_file = open(train_txt_paths, 'w')
    val_image_list_file = open(val_txt_paths, 'w')
    
    validation_video_names = read_validation_video_names(val_video_names_path)


    for video_path in os.listdir(videos_path):
        if '.ts' in video_path:
            if max_vids is not None:
                if vids >= max_vids:
                    break
            video_name = video_path.split('.')[0]
            annotation_name = ''
            for annotation_path in os.listdir(labels_path):
                if video_name in annotation_path:
                    annotation_name = annotation_path
                    break
            print("Total frames: {}".format(frames))
            if annotation_name == '':
                print("Annotations not found for {}".format(video_path))
                continue
            
            annotation_objects = load_annotation_objects(os.path.join(labels_path, annotation_name))

            print("Loading video: {}".format(os.path.join(videos_path, video_path)))
            video = cv2.VideoCapture(os.path.join(videos_path, video_path))

            frame_num = 0 # local frame count (for one video)

            while True:

                if max_frames is not None:
                    if frame_num >= max_frames * FRAME_STEP:
                        break

                return_value, frame = video.read()

                if not return_value:
                    break

                if frame_num % FRAME_STEP == 0:

                    frame_name = str(frames) + '.jpg '

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]

                    frame_objects = get_objects_in_frame(annotation_objects, frame_num)
                    frame_bboxes = get_frame_bboxes(frame_objects)
            
                    if video_path in validation_video_names:
                        label_file = open(os.path.join(val_path, str(frames) + '.txt'), 'w')
                        
                        
                    else:
                        label_file = open(os.path.join(train_path, str(frames) + '.txt'), 'w')

                    for frame_bbox in frame_bboxes:

                        line = ''

                        label = frame_bbox[1]
                        
                        if label in list(class_dict.keys()):
                            line += class_dict[label] + ' '
                        else:
                            continue

                        coords = frame_bbox[0]
                        
                        ###################################################
                        # ---------- YOUR IMPLEMENTATION HERE ----------- #
                        ###################################################
                        
                        
                        # print(coords)
                        x_center = (coords[0][0] + coords[1][0])//2 / frame_width
                        y_center = (coords[0][1] + coords[1][1])//2 / frame_height
                        # print(x_center, y_center)
                        
                        width = np.abs(coords[0][0] - coords[1][0]) / frame_width
                        height = np.abs(coords[0][1] - coords[1][1]) / frame_height
                        # print(width, height)
                        
                        # str_list = [str(x_center)[], str(y_center), str(width), str(height)]
                        coords_string = "{:6f} {:6f} {:6f} {:6f}\n".format(x_center, y_center, width, height)
                        # coords_string = ' '.join(str_list) + '\n'
                        # print(coords_string)
                        

                        ###################################################
                        # ----------- END YOUR IMPLEMENTATION ----------- #
                        ###################################################
                
                        line += coords_string
                    
                        # print(line)
                    
                        label_file.write(line)

                    label_file.close()

                    frame_image = Image.fromarray(frame) # ensure newline character is removed. OpenCV (used by darknet) cannot handle newline character in filename
                    frame_image_name = frame_name[:-1]

                    if video_path in validation_video_names:
                        frame_image.save(os.path.join(val_path, frame_image_name))
                        train_image_list_file.write(os.path.join(val_path, frame_image_name) + '\n')
                    else:
                        frame_image.save(os.path.join(train_path, frame_image_name))
                        val_image_list_file.write(os.path.join(train_path, frame_image_name) + '\n')

                    frames += 1

                frame_num += 1

            vids += 1


    train_image_list_file.close()
    val_image_list_file.close()
        
    print("Darknet dataset (video validation) generated from {} videos with {} frames.".format(str(vids), str(frames)))    

    
def inspect_darknet_dataset(dataset_path, tests=3):
    """
    Visualize random darknet training/validation images and their corresponding labels.
   
    params:
        dataset_path (string): path to the directory containing .jpg images and 
                               corresponding .txt label files
        tests (int): the number of randomly selected examples to visualize
    """
    ###################################################
    # ---------- YOUR IMPLEMENTATION HERE ----------- #
    ###################################################
    
    # imgs = []
    # labels = []
    
    count = 0
    
    # only take several images
    while count<tests:
        
        # randomly select one file from directory
        img_path = random.choice(os.listdir(dataset_path))
        
        # if it's an image
        if '.jpg' in img_path:
            count += 1
            
            label_path = os.path.join(dataset_path, img_path[:-4]+'.txt')
            
           
            with open(label_path) as f:
                line = f.readlines()[0]

            label = line.split(' ')[0]
            bbox = line.split(' ')[1:]
            bbox[-1] = bbox[-1][:-1]
            
            for i in range(len(bbox)):
                bbox[i] = float(bbox[i])
            
            # convert the coordinates of bbox
            x,y,w,h = bbox
            bbox[0] = x-w/2
            bbox[1] = y+h/2
            bbox[2] = x+w/2
            bbox[3] = y-h/2
            
            bbox = [np.array(bbox)]
                        
            img = cv2.imread(os.path.join(dataset_path, img_path))
            # print(img.shape)
            img = plot_boxes_cv2(img, bbox)
            plt.figure()
            plt.imshow(img[:,:,::-1])
            plt.title(img_path+",  label: "+label)
            
        
            
    ###################################################
    # ----------- END YOUR IMPLEMENTATION ----------- #
    ###################################################