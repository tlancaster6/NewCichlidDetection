"""
unit script for running inference using an already trained model
"""
import os
import time
import pandas as pd
import numpy as np
import torch
import torchvision
from Utils.DataLoader import ConvertDataSet
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from ml_utils import Compose, ToTensor, collate_fn
from Utils.Models import model
from Config.config import ROOT_DIR, TRAINED_MODEL_FILE

class Detector:

    def __init__(self, *args):
        self.model = model
        self.model_path = os.path.join(ROOT_DIR, TRAINED_MODEL_FILE)
        self.initiate_model(self.model, self.model_path)

    def detect(self, img_dir):
        # img_dir = "CichlidDetection/inference_imgs"
        """run detection on the images contained in img_dir
        Args:
            img_dir (str): path to the image directory, relative to data_dir (see FileManager)
        """

        pid = img_dir.split('/')[1]
        img_dir = os.path.join(self.fm.local_files['data_dir'], img_dir)
        assert os.path.exists(img_dir)
        img_files = [os.path.join(img_dir, img_file) for img_file in os.listdir(img_dir)]
        dataset = ConvertDataSet(Compose([ToTensor()]), img_files)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True,
                                collate_fn=collate_fn)
        self.evaluate(dataloader, pid)

    def initiate_model(self, model, model_path):
        """initiate the model, optimizer, and scheduler."""
        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=3)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, name):
        """Evaluate on a dataloader"""
        cpu_device = torch.device("cpu")
        self.model.eval()
        results = {}
        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = self.model(images)
            outputs = [{k: v.to(cpu_device).numpy().tolist() for k, v in t.items()} for t in outputs]
            results.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})
        df = pd.DataFrame.from_dict(results, orient='index')
        index_list = df.index.tolist()
        detect_framefiles = []
        for i in index_list:
            detect_framefiles.append(dataloader.dataset.img_files[i])
        df['Framefile'] = [os.path.basename(path) for path in detect_framefiles]
        df = df[['Framefile', 'boxes', 'labels', 'scores']].set_index('Framefile')

        if 'test' in name:
            df.to_csv(os.path.join(self.fm.local_files['detection_dir'], '{}_detections.csv'.format(name)))
        elif 'vid' in name:
            df.to_csv(os.path.join(self.fm.local_files['detection_dir'], '{}_detections.csv'.format(name)))
        else:
            df.to_csv(os.path.join(self.fm.local_files['detection_dir'], '{}_detections.csv'.format(name)))

        return '{}_detections.csv'.format(name)
    
    # def visualise_results(df):
    
