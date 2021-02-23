import os, random, torch, time, pdb

import pandas as pd
import numpy as np

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from Utils.Model import model
from Utils.DataLoader import JPGLoader
from Utils.utils import Logger,AverageMeter,calculate_accuracy,calculate_accuracy_by_projectID

def collate_fn(batch):
    return tuple(zip(*batch))

class ML_model():
    def __init__(self, results_directory, images_directory, annotated_images_file):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.results_directory = results_directory
        self.images_directory = images_directory
        self.images_dt = pd.read_csv(annotated_images_file, index_col=0)
        self.images_dt = self.images_dt[(self.images_dt.CorrectAnnotation == 'Yes') & (self.images_dt.Sex != 'u') & (self.images_dt.Nfish != 0)]

    def createModel(self):
        print('Creating Model')
        self.model = model
        self.model = self.model.to(self.device)
        #self.model = nn.DataParallel(self.model, device_ids=None)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        
    def splitData(self, mode):
        print('Splitting Data')

        if 'Dataset' not in self.images_dt or self.images_dt.Dataset.dtype == 'float64':
            self.images_dt['Dataset'] = ''
    
        train_cutoff = 0.8
        val_cutoff = 1.0

        for image in set(self.images_dt.Framefile):
            if self.images_dt.loc[self.images_dt.Framefile == image].iloc[0].Dataset in ['Train', 'Validate']:
                continue
            p = random.random()
            try:
                if p<=train_cutoff:
                    self.images_dt.loc[self.images_dt.Framefile == image,'Dataset'] = 'Train'                
                elif p<=val_cutoff:
                    self.images_dt.loc[self.images_dt.Framefile == image,'Dataset'] = 'Validate'
                else:
                    pass
            except ValueError:
                pdb.set_trace()

    def createDataLoaders(self, batch_size, n_threads):
        self.batch_size = batch_size
        self.n_threads = n_threads
        print('Creating Data Loaders')

        self.trainData = JPGLoader(self.images_dt[self.images_dt.Dataset == 'Train'], self.images_directory, augment = True)
        self.valData = JPGLoader(self.images_dt[self.images_dt.Dataset == 'Validate'], self.images_directory, augment = True)

        # Output data on split
        self.images_dt.to_csv('FramesSplit.csv', sep = ',')

        self.trainLoader = torch.utils.data.DataLoader(self.trainData, batch_size = 2, shuffle = True, num_workers = self.n_threads, collate_fn=collate_fn)
        self.valLoader = torch.utils.data.DataLoader(self.valData, batch_size = 2, shuffle = False, num_workers = self.n_threads, collate_fn=collate_fn)
        
        print('Done')

    def trainModel(self, n_epochs, nesterov, dampening, learning_rate, momentum, weight_decay, lr_patience):
        
        self.train_logger = Logger(os.path.join(self.results_directory, 'train.log'), ['epoch', 'loss_total', 'loss_classifier','loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg', 'lr'])
        self.train_batch_logger = Logger(os.path.join(self.results_directory, 'train_batch.log'), ['epoch', 'batch', 'iter', 'loss_total', 'lr'])
        #val_logger = Logger(os.path.join(self.results_directory, 'val.log'), ['epoch', 'loss', 'acc'])

        #if nesterov:
         #   dampening = 0
        #else:
        #    dampening = dampening
       
        optimizer = torch.optim.SGD(self.parameters, lr=0.005,momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for i in range(n_epochs + 1):
            loss = self.train_one_epoch(i, self.trainLoader, self.model, optimizer)
            lr_scheduler.step(loss)

            self.val_epoch(i, self.valLoader, self.model)

    def predictLabels(self, trainedModel):
        val_logger = Logger(os.path.join(self.results_directory, 'val.log'), ['epoch', 'loss', 'acc'])

        checkpoint = torch.load(trainedModel)
        begin_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        
        validation_loss,confusion_matrix,_ = self.val_epoch(i, self.valLoader, self.model, self.criterion, val_logger)

        confusion_matrix_file = os.path.join(self.results_directory,'epoch_{epoch}_confusion_matrix.csv'.format(epoch=i))
        confusion_matrix.to_csv(confusion_matrix_file)

    def train_one_epoch(self, epoch, data_loader, model, optimizer):
        print('train at epoch {}'.format(epoch))
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_types = ['loss_total', 'loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
        loss_meters = {loss_type: AverageMeter() for loss_type in loss_types}
        end_time = time.time()



        #batch_time = AverageMeter()
        #data_time = AverageMeter()
        #losses = AverageMeter()
        #accuracies = AverageMeter()

        #end_time = time.time()
        for i, (images, targets) in enumerate(data_loader):

            data_time.update(time.time() - end_time)

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            loss_meters['loss_total'].update(losses.item(), len(images))
            for key, val in loss_dict.items():
                loss_meters[key].update(val.item(), len(images))


            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            self.train_batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss_total': loss_meters['loss_total'].val,
                'lr': optimizer.param_groups[0]['lr']
            })

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=loss_meters['loss_total']))
        self.train_logger.log({
            'epoch': epoch,
            'loss_total': loss_meters['loss_total'].avg,
            'loss_classifier': loss_meters['loss_classifier'].avg,
            'loss_box_reg': loss_meters['loss_box_reg'].avg,
            'loss_objectness': loss_meters['loss_objectness'].avg,
            'loss_rpn_box_reg': loss_meters['loss_rpn_box_reg'].avg,
            'lr': optimizer.param_groups[0]['lr']
        })
        return loss_meters['loss_total'].avg

    def val_epoch(self, epoch, data_loader, model):
        print('validation at epoch {}'.format(epoch))
        model.eval()
        results = {}
        for i, (images, targets) in enumerate(data_loader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            self.calculate_accuracy(targets,outputs)

            #outputs = [{k: v.to(cpu_device).numpy().tolist() for k, v in t.items()} for t in outputs]
            #results.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})

    def calculate_accuracy(self, targets, outputs, conf_cutoff = 0.5):
        for output,target in zip(outputs,targets):
            for score1,box1 in zip(output['scores'],output['boxes']):
                if score1 > conf_cutoff:
                    for score2,box2 in zip(output['scores'],output['boxes']):
                        pdb.set_trace()
                        if box1 != box2 and score2 > conf_cutoff and ret_IOU(box1, box2) > 0:
                            pdb.set_trace()

    def ret_IOU(self, box1, box2):
        overlap_x0, overlap_y0, overlap_x1, overlap_y1 = max(box1[0],box2[0]), max(box1[1],box2[1]), min(box1[2],box2[2]), min(box1[3], box2[3])

        if overlap_x1 < overlap_x0 or overlap_y1 < overlap_y0:
            return(0)
        else:
            intersection = (overlap_x1 - overlap_x0)*(overlap_y1 - overlap_y0)
            union = (box1[2] - box1[0]) * (box1[3] - box1[1]) +  (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
            return(intersection/union)

    