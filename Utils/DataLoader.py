import random, pdb

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as F
from Utils.ConfigurationLoader import Environment
from Utils.FileManager import FileManager
from torch.utils.data import Dataset
import os


def load_boxed_annotation_data(env: Environment, download=True):
    """
    Retrieves annotation data and filters out invalid data
    @param env: Environment file
    @return: data
    """
    fm = FileManager(env)
    if download:
        fm.download_data(env.annotated_data_list)
        fm.download_data(env.annotated_data_folder)
    df = pd.read_csv(fm.map_relative_path_to_local(env.annotated_data_list), index_col=0)
    df = df.loc[df['CorrectAnnotation'] == 'Yes']
    df = df.loc[df['Box'].notna()]
    return df


class BoxedImageLoader(Dataset):
    file_paths: np.ndarray
    labels: np.ndarray
    transforms: list
    """
    Characterizes BoxedImage dataset for torch
    """

    def __init__(self, env: Environment, boxed_images_df: pd.DataFrame, transform=None, train_test_split=0.8):
        boxed_images_df = boxed_images_df[['ProjectID', 'Framefile', 'Sex', 'Box']]
        data: pd.Series = boxed_images_df.groupby(['ProjectID', 'Framefile'])['Box'].apply(list)
        self.env = env
        self.file_paths, self.labels = data.index.to_numpy(), data.values
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        label = self.labels[index]
        fm = FileManager(self.env)
        file_path = fm.map_relative_path_to_local(
            os.path.join(self.env.annotated_data_folder, self.file_paths[index][0],
                         self.file_paths[index][1]))

        image = Image.open(file_path)
        if self.transform is not None:
            image = self.transform(image)
        return torch.tensor(image), torch.tensor(eval(label[0]))


class JPGLoader(object):
    """Class to handle loading of training or testing data"""

    def __init__(self, annotation_dt, jpg_folder, augment):
        """initialize DataLoader
        Args:
            transforms: Composition of Pytorch transformations to apply to the data when loading
            subset (str): data subset to use, options are 'train' and 'test'
        """
        self.ann_dt = annotation_dt
        self.jpg_folder = jpg_folder
        self.augment = augment

        self.images = sorted(set(self.ann_dt.Framefile))
        random.shuffle(self.images)

        # self.target_transforms = {'m':1,'f':2, 'u':3}
        self.target_transforms = ['b', 'm', 'f', 'u']

    def __getitem__(self, idx):
        """get the image and target corresponding to idx
        Args:
            idx (int): image ID number, 0 indexed
        Returns:
            tensor: img, a tensor image
            dict of tensors: target, a dictionary containing the following
                'boxes', a size [N, 4] tensor of target annotation boxes
                'labels', a size [N] tensor of target labels (one for each box)
                'image_id', a size [1] tensor containing idx
        """
        # read in the image and label corresponding to idx
        img_file = self.jpg_folder + self.images[idx]
        img = Image.open(img_file).convert("RGB")

        # grab annotations for that image
        annotations = self.ann_dt[self.ann_dt.Framefile == self.images[idx]]
        N_ann = annotations.iloc[0].Nfish  # Number of annotations

        boxes = []
        labels = []
        area = []
        image_id = [idx]

        if N_ann == 0:
            # pick random box
            labels = [[0]]
        else:
            for index, row in annotations.iterrows():
                (x1, y1, w, h) = eval(row.Box)
                boxes.append((x1, y1, x1 + w, y1 + h))
                labels.append(self.target_transforms.index(row.Sex))
                area.append(w * h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.int64)
        image_id = torch.as_tensor(image_id, dtype=torch.int64)

        return F.to_tensor(img), {'boxes': boxes, 'labels': labels, 'area': area, 'image_id': image_id}

    def __len__(self):
        return len(self.images)


class VideoLoader(object):

    def __init__(self, videofile, framelist):
        self.videofile = videofile
        self.framelist = framelist

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.videofile)
        frame_number = self.framelist[idx]
        cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_number - 1)
        res, frame = cap.read()
