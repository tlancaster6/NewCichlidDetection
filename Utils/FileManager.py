#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
from Utils.ConfigurationLoader import Environment
from Config.config import ROOT_DIR_
import os
import pdb


class FileManager:

    def __init__(self, env: Environment):
        """
        Class for handling file operations (uploads and downloads, local read and write, etc.)
        """
        self.local_master_dir = env.local_file_path_root
        self.cloud_master_dir = env.cloud_file_path_root
        self.project_id = env.project_id
        self.rclone_config = env.rclone_config

    def map_relative_path_to_local(self, relative_path: str):
        """
        maps relative data file path to local storage path
        @param relative_path:
        @return:
        """
        return os.path.join(os.path.join(ROOT_DIR_, self.local_master_dir), relative_path)

    def map_relative_path_to_cloud(self, relative_path: str):
        """
        maps relative data file path to cloud path
        @param relative_path:
        @return: cloud path directory, file name
        """
        return os.path.join(self.cloud_master_dir, relative_path)

    def check_file_exists(self, relative_path):
        """
        check if a local file exists in corresponding dropbox storage location
        @param relative_path: relative path to file
        @return: True if file exists in
        """
        cloud_path, relative_name = os.path.split(self.map_relative_path_to_cloud(relative_path))

        output = subprocess.run(['rclone', 'lsf', cloud_path], capture_output=True, encoding='utf-8')
        remote_files = [x.rstrip('/') for x in output.stdout.split('\n')]

        return relative_name in remote_files

    def create_directory(self, directory):
        """
        creates a directory in local data directory
        @param directory: directory to add
        @return: None
        """
        new_dir = self.local_master_dir + directory
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    @staticmethod
    def untar_file(local_path, relative_name):
        """
        Given a file directory and file name, untar the file and remove the original .tar
        @param local_path: directory containing file
        @param relative_name: file name
        @return:
        """
        # Untar directory
        tarred = relative_name.endswith('.tar')
        if tarred:
            output = subprocess.run(['tar', '-xvf', os.path.join(local_path, relative_name), '-C', local_path],
                                    capture_output=True,
                                    encoding='utf-8')
            if output.stderr != '':
                print(output.stderr)
            output = subprocess.run(['rm', '-f', os.path.join(local_path, relative_name)], capture_output=True,
                                    encoding='utf-8')
            if output.stderr != '':
                print(output.stderr)

    def download_data(self, relative_path, untar=True):
        """
        download data from dropbox
        @param untar:
        @param relative_path: relative path of data
        @return:
        """
        assert relative_path is not None, "relative_path was None"

        local_path_dir, name = os.path.split(self.map_relative_path_to_local(relative_path))
        cloud_path_dir, name = os.path.split(self.map_relative_path_to_cloud(relative_path))

        # pull files from cloud path to local path directory
        output = subprocess.run(
            ['rclone', 'copy', f'{self.rclone_config}:{os.path.join(cloud_path_dir, name)}', local_path_dir],
            capture_output=True, encoding='utf-8')

        if output.stderr != '':
            print(f'There was an error downloading data {relative_path}. {output.stderr}')

        # untar files if requested
        if untar:
            # check if specified document is a file or a directory
            if os.path.isfile(os.path.join(local_path_dir, name)):
                # if document is a file untar it
                self.untar_file(local_path_dir, name)

            else:
                # if document is a directory, traverse and untar each file
                path = os.path.join(local_path_dir, name)
                for f in os.listdir(path):
                    self.untar_file(path, f)

    def upload_data(self, local_data, tarred = False):
        
        assert local_data is not None, "Local_data was None"

        local_path_dir, name = os.path.split(self.map_relative_path_to_local(local_data))
        cloud_path_dir, name = os.path.split(self.map_relative_path_to_cloud(local_data))

        relative_name = local_data.rstrip('/').split('/')[-1]

        if tarred:
            output = subprocess.run(['tar', '-cvf', local_path_dir + relative_name + '.tar', '-C', local_path_dir, relative_name], capture_output = True, encoding = 'utf-8')
            if output.returncode != 0:
                print(output.stderr)
                raise Exception('Error in tarring ' + local_data)
            relative_name += '.tar'

        if os.path.isdir(local_path_dir + relative_name):
            output = subprocess.run(['rclone', 'copy', local_path_dir + relative_name, cloud_path_dir + relative_name], capture_output = True, encoding = 'utf-8')
            #subprocess.run(['rclone', 'check', local_path_dir + relative_name, cloud_path_dir + relative_name], check = True) #Troubleshooting directory will have depth data in it when you upload the cluster data

        elif os.path.isfile(local_path_dir + relative_name):
            print(['rclone', 'copy', local_path_dir + relative_name, cloud_path_dir])
            output = subprocess.run(['rclone', 'copy', local_path_dir + relative_name, cloud_path_dir], capture_output = True, encoding = 'utf-8')
            output = subprocess.run(['rclone', 'check', local_path_dir + relative_name, cloud_path_dir], check = True, capture_output = True, encoding = 'utf-8')
        else:
            raise Exception(local_data + ' does not exist for upload')

        if output.returncode != 0:
            pdb.set_trace()
            raise Exception('Error in uploading file: ' + output.stderr)

    def upload_data_and_merge(self):
        # TODO: finish upload data and merge
        pass
