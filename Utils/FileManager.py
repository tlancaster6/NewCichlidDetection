#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, subprocess, pdb, platform, shutil
from Utils.ConfigurationLoader import Environment

current_directory = os.path.dirname(os.path.realpath(__file__))


class FileManager:

    def __init__(self, env: Environment):
        """
        Class for handling file operations (uploads and downloads, local read and write, etc.
        """
        self.local_master_dir = env.local_file_path_root
        self.cloud_master_dir = env.cloud_file_path_root
        self.project_id = env.project_id

    def map_local_path_to_cloud(self, local_file_path: str):
        """
        converts a local file path and maps it to corresponding cloud file path
        @param local_file_path:
        @return: cloud path directory, file name
        """
        relative_name = local_file_path.rstrip('/').split('/')[-1]
        local_path = local_file_path.split(relative_name)[0]
        cloud_path = local_path.replace(self.local_master_dir, self.cloud_master_dir)
        return cloud_path, local_path, relative_name

    def check_file_exists(self, local_file_path):
        """
        check if a local file exists in corresponding dropbox storage location
        @param local_file_path: path to file
        @return: True if file exists in
        """
        cloud_path, local_path, relative_name = self.map_local_path_to_cloud(local_file_path)

        output = subprocess.run(['rclone', 'lsf', cloud_path], capture_output=True, encoding='utf-8')
        remote_files = [x.rstrip('/') for x in output.stdout.split('\n')]

        if relative_name in remote_files:
            return True
        else:
            return False

    def create_directory(self, directory):
        """
        creates a directory in local data directory
        @param directory: directory to add
        @return: None
        """
        new_dir = self.local_master_dir + directory
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    def download_data(self, local_file_path, tarred=False, tarred_subdirs=False, allow_errors=False, quiet=False):
        """
        downloads specific file from dropbox folder
        @param local_file_path:
        @param tarred:
        @param tarred_subdirs:
        @param allow_errors: 
        @param quiet:
        @return:
        """

        assert local_file_path is not None, "local_file_path was None"

        cloud_path, local_path, relative_name = self.map_local_path_to_cloud(local_file_path)
        relative_name += '.tar' if tarred else ''

        # list files in corresponding cloud directory
        cloud_objects = subprocess.run(['rclone', 'lsf', cloud_path], capture_output=True,
                                       encoding='utf-8').stdout.split()

        if relative_name + '/' in cloud_objects:  # directory
            output = subprocess.run(['rclone', 'copy', cloud_path + relative_name, local_path + relative_name],
                                    capture_output=True, encoding='utf-8')
        elif relative_name in cloud_objects:  # file
            output = subprocess.run(['rclone', 'copy', cloud_path + relative_name, local_path], capture_output=True,
                                    encoding='utf-8')
        else:
            if allow_errors:
                if not quiet:
                    print('Warning: Cannot find {}. Continuing'.format(cloud_path + relative_name))
                else:
                    pass
            else:
                raise FileNotFoundError('Cant find file for download: ' + cloud_path + relative_name)

        if not os.path.exists(local_path + relative_name):
            if allow_errors:
                if not quiet:
                    print('Warning. Cannot download {}. Continuing'.format(local_path + relative_name))
                else:
                    pass
            else:
                raise FileNotFoundError('Error downloading: ' + local_path + relative_name)

        if tarred:
            # Untar directory
            output = subprocess.run(['tar', '-xvf', local_path + relative_name, '-C', local_path], capture_output=True,
                                    encoding='utf-8')
            output = subprocess.run(['rm', '-f', local_path + relative_name], capture_output=True, encoding='utf-8')

        if tarred_subdirs:
            for d in [x for x in os.listdir(local_file_path) if '.tar' in x]:
                output = subprocess.run(
                    ['tar', '-xvf', local_file_path + d, '-C', local_file_path, '--strip-components', '1'],
                    capture_output=True, encoding='utf-8')
                os.remove(local_file_path + d)

    def upload_data(self, local_data, tarred=False):

        relative_name = local_data.rstrip('/').split('/')[-1]
        local_path = local_data.split(relative_name)[0]
        cloud_path = local_path.replace(self.local_master_dir, self.cloud_master_dir)

        if tarred:
            output = subprocess.run(
                ['tar', '-cvf', local_path + relative_name + '.tar', '-C', local_path, relative_name],
                capture_output=True, encoding='utf-8')
            if output.returncode != 0:
                print(output.stderr)
                raise Exception('Error in tarring ' + local_data)
            relative_name += '.tar'

        if os.path.isdir(local_path + relative_name):
            output = subprocess.run(['rclone', 'copy', local_path + relative_name, cloud_path + relative_name],
                                    capture_output=True, encoding='utf-8')
            # subprocess.run(['rclone', 'check', local_path + relative_name, cloud_path + relative_name], check = True) #Troubleshooting directory will have depth data in it when you upload the cluster data

        elif os.path.isfile(local_path + relative_name):
            print(['rclone', 'copy', local_path + relative_name, cloud_path])
            output = subprocess.run(['rclone', 'copy', local_path + relative_name, cloud_path], capture_output=True,
                                    encoding='utf-8')
            output = subprocess.run(['rclone', 'check', local_path + relative_name, cloud_path], check=True,
                                    capture_output=True, encoding='utf-8')
        else:
            raise Exception(local_data + ' does not exist for upload')

        if output.returncode != 0:
            pdb.set_trace()
            raise Exception('Error in uploading file: ' + output.stderr)

    def upload_and_merge(self, local_data, master_file, tarred=False, ID=False):
        if os.path.isfile(local_data):
            # We are merging two crv files
            self.download_data(master_file)
            import pandas as pd
            if ID:
                old_dt = pd.read_csv(master_file, index_col=ID)
                new_dt = pd.read_csv(local_data, index_col=ID)
                old_dt = old_dt.append(new_dt)
                old_dt.index.name = ID
            else:
                old_dt = pd.read_csv(master_file)
                new_dt = pd.read_csv(local_data)
                old_dt = old_dt.append(new_dt)

            old_dt.to_csv(master_file, sep=',')
            self.upload_data(master_file)
        else:
            # We are merging two tarred directories
            try:
                self.download_data(master_file, tarred=True)
            except FileNotFoundError:
                self.create_directory(master_file)
            for nfile in os.listdir(local_data):
                subprocess.run(['mv', local_data + nfile, master_file])
            self.upload_data(master_file, tarred=True)
