"""
This file contains classification and regression dataset classes that inherit
from the torch.utils.data.Dataset class.

E6692 Spring 2022
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
import glob
import os
import uuid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


DEFAULT_TRANSFORMS = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ClassificationDataset(Dataset):
    """
    A torch.utils.data.Dataset class for implementing Classification
    dataset functionality.

    Methods:
        __len__() : return the length of the dataset

        __getitem__(index) : return the image and class index of the image
                             specified by 'index'

        _refresh() : load the annotations upon dataset initialization

        save_image(image, class_name) : makes an entry into the dataset

        get_count(class_name) : get number of entries with class_name

        get_random_image() : returns a random image and corresponding class
                             index from the dataset

        grid_visualization(grid_dims) : generates a grid visualization
                                        of dataset images.
    """

    def __init__(self, path, class_names, dataset_name, transforms=None):
        """
        Initialize the ClassificationDataset.

        params:
            path (string): path directory containing the dataset images.
            class_names (list of strings): class names.
            dataset_name (string): name of the dataset
            transforms=None : optional, instance of torchvision.transforms.Compose
        """
        self.path = path
        self.classes = class_names
        self.dataset_name = dataset_name
        self.transforms = transforms
        self._refresh() # load the images


    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.annotations)


    def __getitem__(self, index):
        """
        Return the image and class index of the image specified by 'index'.

        This function will be used by torch.utils.data.DataLoader.

        Note: If transforms=None, be sure to apply DEFAULT_TRANSFORMS

        param:
            index (int): the index of the image to access. Ranges from 0 to __len__()

        returns:
            image (torch.Tensor): PyTorch tensor representation of the image
            annotation (int): the class index of the image label
        """
        annotation = self.annotations[index]

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.ClassificationDataset.__getitem__() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return image, annotation['class_index']


    def _refresh(self):
        """
        Loads the annotations upon dataset initialization. This function populates
        the list self.annotations with a dictionary entry for each annotated image.
        Each element of self.annotations is a dictionary of the following form:

        self.annotations[index] --> {'path' : path to the image file (string),
                                     'class_index' : the index of the class (int),
                                     'class_name' : the name of the class (string)}
        """

        self.annotations = []

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.ClassificationDataset._refresh() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################


    def save_image(self, image, class_name):
        """
        Save an image of the specified class to self.path. For each class there
        should be a corresponding class image directory. For example, if you have
        two classes, left and right, then your directory structure should look
        like this:

        self.path
        |--left
        |   |--image_name.jpg
        |   |--image_name2.jpg
        |   |-- ...
        |
        |--right
        |   |--image_name.jpg
        |   |--image_name2.jpg
        |   | -- ...
        ...

        Use uuid.uuid1() to generate image filenames, and call self._refresh()
        after sucessfully saving an image to the dataset.

        Don't forget to call self._refresh() when finished saving the image.

        params:
            image (np.array): numpy array representation of the image.
            class_name (string): the corresponding class label.

        returns:
            image_path (string): the path to the saved image (including the filename).
        """
        if class_name not in self.classes:
            raise Exception('"{}" is not a defined class.'.format(class_name))

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.ClassificationDataset.save_image() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return image_path


    def get_count(self, class_name):
        """
        Returns the number of entries of class_name.

        param:
            class_name (string): name of the class

        returns:
            count (int): the number of images saved as class_name
        """
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.ClassificationDataset.get_count() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return count


    def get_random_image(self):
        """
        Get a random image from the dataset. This returns a random image from
        the dataset. The class the image is chosen from is also random, and an
        exception is thrown if the dataset is empty.

        returns:
            image (np.array): numpy array representation of the image
            annotation (int): the class index of the image label
        """
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.ClassificationDataset.get_random_image() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return image, annotation['class_index']


    def grid_visualization(self, grid_dims=(5, 5)):
        """
        Creates a grid visualization of the images in the dataset. The visualization
        is titled with the name of the dataset and the dimensions of the
        visualization are determined by grid_dims. The visualization is displayed
        in the cell output of the cell that this function is called in with plt.show().

        param:
            grid_dims (tuple of ints): the dimensions of the visualization (2D only).
        """
        cols = []
        for _ in range(grid_dims[0]):
            row = []
            for _ in range(grid_dims[1]):
                row.append(self.get_random_image()[0])
            cols.append(np.hstack(row))
        grid = np.vstack(cols)

        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis('off')
        plt.title(self.dataset_name)
        plt.show()



class RegressionDataset(Dataset):
    """
    A torch.utils.data.Dataset class for implementing Regression
    dataset functionality.

    Methods:
        __len__() : return the length of the dataset

        __getitem__(index) : return the image, regression class index, and
                             regression point coordinates of the image specified
                             by 'index'

        _parse(path) : get regression point coordinates from image filename

        refresh() : load the annotations upon dataset initialization

        save_image(image, class_name) : makes an entry into the dataset

        get_count(class_name) : get number of entries with class_name

        get_random_image() : returns a random image and corresponding class
                             index from the dataset

        grid_visualization(grid_dims) : generates a grid visualization
                                        of dataset images.

        populate_from_files(unlabeled_image_names, unlabeled_data_path, labels_path) :
                    populates the regression dataset by taking the unlabeled
                    images and pairing them with their corresponding labels
    """

    def __init__(self, path, regression_categories, dataset_name, transforms=None):
        """
        Initialize the RegressionDataset.

        params:
            path (string): path directory containing the dataset images.
            regression_categories (list of strings): regression class names.
            dataset_name (string): name of the dataset
            transforms=None : optional, instance of torchvision.transforms.Compose
        """
        self.path = path
        os.system('rm -r {}'.format(path))
        self.regression_categories = regression_categories
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.refresh()


    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.annotations)


    def __getitem__(self, index):
        """
        Return the image, class index, and regression coordinates of the image
        specified by 'index'. A transformation is applied to the coordinates
        before being returned to . The coordinate transformation is:

        x = 2 * (x / image_width - 1/2)
        y = 2 * (y / image_height - 1/2)

        This function will be used by torch.utils.data.DataLoader.

        Note: If transforms=None, be sure to apply DEFAULT_TRANSFORMS

        param:
            index (int): the index of the image to return

        returns:
            image (torch.Tensor): PyTorch tensor representation of the image
            regression_class_index (int): the index of the corresponding
                                          regression class
            coordinates (torch.Tensor): length 2 torch Tensor for the
                                        coordinates of the regression point
        """
        annotation = self.annotations[index]

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.RegressionDataset.__getitem__() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return image, annotation['regression_category_index'], torch.Tensor([x, y])


    def _parse(self, path):
        """
        Regression points are stored in the filename of the image. For example,
        "100_200_notarealimage.jpg" indicates that in the image the keypoint
        is at (100, 200) measured in pixels. This function returns only the
        coordinates (x, y).

        params:
            path (string): name of the stored image. See above for example.

        returns:
            coordinates (tuple of ints): length 2 tuple of regression point coordinates
        """
        image_name = os.path.basename(path)
        values = image_name.split('_')

        coordinates = int(values[0]), int(values[1])

        return coordinates


    def refresh(self):
        """
        Loads the annotations upon dataset initialization. This function populates
        the list self.annotations with a dictionary entry for each annotated image.
        Each element of self.annotations is a dictionary of the following form:

        self.annotations[index] --> {'image_path' : path to the image file (string),
                                     'regression_category_index' : the index of the regression class (int),
                                     'regression_category' : the name of the regression class (string)}
                                     'x' : the X coordinate of the regression point (int),
                                     'y' : the Y coordinate of the regression point (int)
                                     }
        """
        self.annotations = []

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.RegressionDataset.refresh() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################


    def save_image(self, regression_category, image, x, y):
        """
        Save an image of the specified regression class to self.path. For each class there
        should be a corresponding class image directory. Filenmes should include the regression
        point coordinates separated by a '_'. For example, if you have
        two regression classes, left ear and right ear, then your directory structure should look
        like this:

        self.path
        |--left ear
        |   |--x1_y1_image_name.jpg
        |   |--x2_y2_image_name2.jpg
        |   |-- ...
        |
        |--right ear
        |   |--x1_y1_image_name.jpg
        |   |--x2_y2_image_name2.jpg
        |   | -- ...
        ...

        Use uuid.uuid1() to generate image filenames, and call self._refresh()
        after sucessfully saving an image to the dataset.

        Don't forget to call self.refresh() when finished saving the image.

        params:
            regression_category (string): the name of the regression category
            image (np.array): numpy array representation of the image.
            x (int): the X coordinate of the regression point
            y (int): the Y coordinate of the regression point

        returns:
            image_path (string): the path to the saved image (including the filename).
        """
        if regression_category not in self.regression_categories:
            raise Exception('"{}" is not a defined regression class.'.format(regression_category))

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.RegressionDataset.save_image() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return image_path


    def get_count(self, regression_category):
        """
        Returns the number of entries of class_name.

        param:
            class_name (string): name of the class

        returns:
            count (int): the number of images saved as class_name
        """
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.RegressionDataset.get_count() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return count


    def get_random_image(self):
        """
        This function returns a random image from the dataset. The class the
        image is chosen from is also random, and an exception is thrown if
        the dataset is empty.

        returns:
            image (np.array): numpy array representation of the image
            annotation (int): the regression class index of the labeled regression point
            coordinates (tuple of ints): length 2 -> (x, y)
        """
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.RegressionDataset.get_random_image() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return image, annotation['regression_category_index'], (annotation['x'], annotation['y'])


    def grid_visualization(self, grid_dims=(5, 5)):
        """
        Creates a grid visualization of the images in the dataset. The visualization
        is titled with the name of the dataset and the dimensions of the
        visualization are determined by grid_dims. The visualization is displayed
        in the cell output of the cell that this function is called in with plt.show().

        param:
            grid_dims (tuple of ints): the dimensions of the visualization (2D only).
        """
        cols = []
        for _ in range(grid_dims[0]):
            row = []
            for _ in range(grid_dims[1]):
                image, _, coords = self.get_random_image()
                image = cv2.circle(image, coords, 10, (0, 255, 0), 2)
                row.append(image)
            cols.append(np.hstack(row))
        grid = np.vstack(cols)

        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis('off')
        plt.title(self.dataset_name)
        plt.show()


    def populate_from_files(self, unlabeled_image_names, unlabeled_data_path, labels_path):
        """
        This function populates the regression dataset by taking the unlabeled
        images and pairing them with their corresponding labels. The function
        get_label_dict() is called to read the text file labels as a dictionary.

        params:
            unlabeled_image_names (list of strings): names of the unlabeled images.
            unlabeled_data_path (string): path to the unlabeled images.
            labels_path (string): path to the labels file (regression_labels.txt)
        """
        os.system('rm -r {}'.format(self.path))

        label_dict = get_label_dict(labels_path, self.regression_categories)

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.datasets.RegressionDataset.get_random_image() not implemented!') # delete me

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################


def get_label_dict(labels_path, regression_classes):
    """
    Reads the labels text file in as a dictionary 'label_dict'. Each element of
    label_dict is of the form:

    label_dict[file_name] --> { regression_class_1 : (x, y),
                                regression_class_2 : (x, y),}
                                regression_class_3 : (x, y),
                                        ...
                               }

    The expected regression label format in the labels text file by line is:

    image_filename CAT1_X CAT1_Y CAT2_X CAT2_Y CAT3_X CAT3_Y ...

    params:
        labels_path (string): path to the text file containing regression labels.
        regression_classes (list of strings): names of the regression classes.

    returns:
        label_dict (dict of dicts): see format description above
    """
    label_dict = {}

    with open(labels_path, 'r') as labels:
        lines = labels.readlines()[1:]
        for line in lines:
            values = line.split(' ')
            num_values = len(values)
            if num_values > 1:
                file_name = values[0]
                label_dict.update({ file_name : {} })
                for i in range(len(regression_classes)):
                    try:
                        cat_x = int(values[2 * i + 1])
                        cat_y = int(values[2 * i + 2])
                        label_dict[file_name].update({regression_classes[i] : (cat_x, cat_y) })
                    except:
                        raise Exception('Labels file not formatted correctly.')

    return label_dict
