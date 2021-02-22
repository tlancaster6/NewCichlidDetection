import random, pdb
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as F

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

        self.target_transforms = {'m':1,'f':2, 'u':3}

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
        N_ann = annotations.iloc[0].Nfish # Number of annotations

        boxes = []
        labels = []
        area = []
        image_id = [idx]

        if N_ann == 0:
            #pick random box
            labels = [[0]]
        else:
            for index, row in annotations.iterrows():
                (x1,y1,w,h) = eval(row.Box)
                boxes.append((x1,y1,x1+w,y1+h))
                labels.append(self.target_transforms[row.Sex])
                area.append(w*h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.int64)
        image_id = torch.as_tensor(image_id, dtype=torch.int64)

        return F.to_tensor(img), {'boxes':boxes, 'labels': labels, 'area':area, 'image_id':image_id}

    def __len__(self):
        return len(self.images)
