import argparse, subprocess, datetime, os, pdb, sys
from Utils.CichlidObjectDetection import ML_model


parser = argparse.ArgumentParser(description='This script takes video clips and annotations, either train a model from scratch or finetune a model to work on the new animals not annotated')
# Input data
parser.add_argument('--Videos_directory', type = str, required = True, help = 'Name of directory that holds mp4 files')
parser.add_argument('--Videos_file', type = str, required = True, help = 'Csv file listing each video. Must contain three columns: VideoFile, Label, ProjectID')

# Output data
parser.add_argument('--Results_directory', type = str, required = True, help = 'Directory to store output files')

# Dataloader options
parser.add_argument('--n_threads', default=5, type=int, help='Number of threads for multi-thread loading')
parser.add_argument('--batch_size', default=13, type=int, help='Batch Size')
parser.add_argument('--gpu', default='0', type=str, help='The index of GPU to use for training')

# Parameters specific for training from scratch
parser.add_argument('--n_epochs',default=100,type=int, help='Number of total epochs to run')

# Parameters specific for finetuning for other animals

args = parser.parse_args()

#subprocess.call(['rclone', 'copy', 'cichlidVideo:McGrath/Apps/CichlidPiData/__AnnotatedData/BoxedFish/', 'AnnotatedData'])
#print()
#for d in [x for x in os.listdir('AnnotatedData/BoxedImages/') if '.tar' in x]:
	#print(' '.join(['tar', '-xvf', d, '-C', 'AnnotatedData/BoxedImages/', '--strip-components', '1']))
#	subprocess.call(['tar', '-xvf', 'AnnotatedData/BoxedImages/' + d, '-C', 'AnnotatedData/BoxedImages/', '--strip-components', '1'])


# Make directory if necessary
if not os.path.exists(args.Results_directory):
    os.makedirs(args.Results_directory)

with open(os.path.join(args.Results_directory, 'TrainingLog.txt'),'w') as f:
	for key, value in vars(args).items():
		print(key + ': ' + str(value), file = f)
	print('PythonVersion: ' + sys.version.replace('\n', ' '), file = f)
	import pandas as pd
	print('PandasVersion: ' + pd.__version__, file = f)
	import numpy as np
	print('NumpyVersion: ' + np.__version__, file = f)
	import torch
	print('pytorch: ' + torch.__version__, file = f)
	
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
learningObj = ML_model(args.Results_directory, args.Videos_directory, args.Videos_file)
learningObj.createModel()
learningObj.splitData('Train')
learningObj.createDataLoaders(args.batch_size, args.n_threads)
learningObj.trainModel(args.n_epochs)
