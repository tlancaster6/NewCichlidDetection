import csv, pdb
import pandas as pd
import scipy.special
import argparse, os, cv2
from collections import namedtuple



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    try:
        n_correct_elems = correct.float().sum().data[0]
    except:
        n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def calculate_accuracy_by_projectID(outputs, targets, videofile, projectID):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1)).cpu()

    confidence = scipy.special.softmax(outputs.cpu().detach().numpy(), axis=1).max(axis=1)

    return pd.DataFrame({'VideoFile': videofile, 'ProjectID': projectID, 'Correct': correct[0], 'Predictions': pred[0].cpu(), 'Confidence':confidence})


def vid_to_imgs(videofile, projectID, frame_step, resultdir):
    localpath = os.path.dirname(__file__)
    targetpath = os.path.join(os.path.dirname(__file__),"Input",videofile)
    # print (os.path.exists(targetpath))

    vidcap = cv2.VideoCapture(targetpath)
    print("Total Frames: ",vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success,image = vidcap.read()    
    count = 0
    while success:

        savedfile = str(projectID +"_"+ videofile + "_" + "frame%d.jpg" % (count*frame_step))
        cv2.imwrite(os.path.join(localpath,resultdir,savedfile), image)     # save frame as JPEG file      
        print('Read a new frame: ', success)

        count += 1
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, count*frame_step)
        success,image = vidcap.read()
        


parser = argparse.ArgumentParser(description='This script takes video clips and annotations, either train a model from scratch or finetune a model to work on the new animals not annotated')
# Input data
parser.add_argument('--Video_file', type = str, required = True, help = 'Path to a video file')
parser.add_argument('--ProjectID', type = str, required = True, help = 'Project ID')
parser.add_argument('--Frame_step', type = int, required = True, help = 'N frame')
# Output data
parser.add_argument('--Results_directory', type = str, required = True, help = 'Image output directory')

args = parser.parse_args()

if not os.path.exists(args.Results_directory):
    os.makedirs(args.Results_directory)

vid_to_imgs(args.Video_file, args.ProjectID, args.Frame_step, args.Results_directory)
