import os.path

from Utils.FileManager import FileManager
from Utils.DataLoader import load_boxed_annotation_data
from Utils.ConfigurationLoader import Environment, load_environment
import pandas as pd
import shutil


def move_images_to_one_folder(env: Environment):
    df: pd.DataFrame = load_boxed_annotation_data(env, download=False)
    fm = FileManager(env)
    df = df.reset_index()
    df = df.drop_duplicates(subset=['Framefile'])
    for row in df.index:
        project = df['ProjectID'][row]
        file_name = df['Framefile'][row]
        local_path = fm.map_relative_path_to_local(os.path.join(env.annotated_data_folder, project, file_name))
        dest_path = fm.map_relative_path_to_local(os.path.join('images/', file_name))
        os.rename(local_path, dest_path)


def move_files_to_folder(files, destination_folder, env: Environment):
    fm = FileManager(env)
    for f in files:
        img = fm.map_relative_path_to_local(os.path.join('images/', f))
        lbl = fm.map_relative_path_to_local(os.path.join('labels/', f.replace('.jpg', '.txt')))
        os.rename(img, fm.map_relative_path_to_local(os.path.join(f'images/{destination_folder}/', f)))
        os.rename(lbl, fm.map_relative_path_to_local(
            os.path.join(f'labels/{destination_folder}/', f.replace('.jpg', '.txt'))))


def partition_dataset(env: Environment):
    df: pd.DataFrame = load_boxed_annotation_data(env, download=False)

    df = df.reset_index()
    df = df.drop_duplicates(subset=['Framefile'])
    df = df.set_index('Framefile')

    p = [0.8, 0.95, 1]
    for i in range(len(p)):
        p[i] *= len(df.index)
        p[i] = int(p[i])
    train = list(df.index[:p[0]])
    test = list(df.index[p[0]:p[1]])
    val = list(df.index[p[1]:p[2]])
    move_files_to_folder(train, 'train', env)
    move_files_to_folder(test, 'test', env)
    move_files_to_folder(val, 'val', env)


def generate_labels(env: Environment):
    df: pd.DataFrame = load_boxed_annotation_data(env, download=False)
    fm = FileManager(env)
    df = df.set_index('Framefile')
    W = 1296
    H = 972
    for row in df.index:
        file_name = row
        file = fm.map_relative_path_to_local(os.path.join('labels/', file_name.replace('.jpg', '.txt')))
        classes = df[df.index == file_name]['Sex']
        box = df[df.index == file_name]['Box']
        annotations = []
        for i in range(len(classes)):
            c = 0 if classes[i] == 'm' else 1
            # convert box to cx, cy, w, h
            try:
                b = eval(box[i])
            except:
                print('hih')
            w, h = b[2], b[3]
            cx, cy = b[0] + w / 2, b[1] + h / 2

            w, h = w / W, h / H
            cx, cy = cx / W, cy / H
            annotations.append(f'{c} {cx} {cy} {w} {h}')
        with open(file, 'w+') as f:
            for annotation in annotations:
                f.write(annotation)
                f.write('\n')
