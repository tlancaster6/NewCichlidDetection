#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
from Utils.ConfigurationLoader import Environment
from Config.config import ROOT_DIR
import os

current_directory = os.path.dirname(os.path.realpath(__file__))


class FileManager:

    def __init__(self, env: Environment):
        """
        Class for handling file operations (uploads and downloads, local read and write, etc.
        """
        self.local_master_dir = env.local_file_path_root
        self.cloud_master_dir = env.cloud_file_path_root
        self.project_id = env.project_id
        self.rclone_config = env.rclone_config

    def map_relative_path_to_local(self, relative_path: str):
        """
        converts a relative file path and maps it to a local file path
        @param relative_path:
        @return:
        """
        return os.path.join(os.path.join(ROOT_DIR, self.local_master_dir), relative_path)

    def map_relative_path_to_cloud(self, relative_path: str):
        """
        converts a relative file path and maps it to corresponding cloud file path
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
        cloud_path, relative_name = self.map_relative_path_to_cloud(relative_path)

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

    def untar_file(self, local_path, relative_name):
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

        local_path, relative_name = os.path.split(self.map_relative_path_to_local(relative_path))
        cloud_path, relative_name = os.path.split(self.map_relative_path_to_cloud(relative_path))

        # pull files from corresponding directory
        output = subprocess.run(['rclone', 'copy', f'{self.rclone_config}:{os.path.join(cloud_path, relative_name)}',
                                 os.path.join(local_path)], capture_output=True, encoding='utf-8')

        if output.stderr != '':
            print(f'There was an error downloading data {relative_path}. {output.stderr}')

        # untar files if requested
        if untar:
            # check if specified document is a file or a directory
            if os.path.isfile(os.path.join(local_path, relative_name)):
                # if document is a file untar if requested
                self.untar_file(local_path, relative_name)

            else:
                # if document is a directory, recursively traverse file directory to untar
                path = os.path.join(local_path, relative_name)
                for f in os.listdir(path):
                    self.untar_file(path, f)
<<<<<<< Updated upstream
=======

    def move_file(self, relative_path_loc: str, relative_path_dest: str):
        os.rename(self.map_relative_path_to_local(relative_path_loc),
                  self.map_relative_path_to_local(relative_path_dest))

    def upload_data(self):
        # TODO: finish upload data
        pass

    def upload_data_and_merge(self):
        # TODO: finish upload data and merge
        pass
>>>>>>> Stashed changes
