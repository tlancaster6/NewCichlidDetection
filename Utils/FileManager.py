#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, subprocess, pdb, platform, shutil
current_directory = os.path.dirname(os.path.realpath(__file__))
class FileManager():
    
    def __init__(self, projectID = None, rcloneRemote = 'cichlidVideo:', masterDir = 'McGrath/Apps/CichlidPiData/'):
        """
        Class for handling file operations (uploads and downloads, local read and write, etc.
        """
        localMasterDir = "NewCichlidDetection"
        
    def checkFileExists(self, local_data):
        relative_name = local_data.rstrip('/').split('/')[-1]
        local_path = local_data.split(relative_name)[0]
        cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)

        output = subprocess.run(['rclone', 'lsf', cloud_path], capture_output = True, encoding = 'utf-8')
        remotefiles = [x.rstrip('/') for x in output.stdout.split('\n')]

        if relative_name in remotefiles:
            return True
        else:
            return False

    
    def createDirectory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    def downloadData(self, local_data, tarred = False, tarred_subdirs = False, allow_errors=False, quiet=False):
        if local_data is None:
            return
        relative_name = local_data.rstrip('/').split('/')[-1] + '.tar' if tarred else local_data.rstrip('/').split('/')[-1]
        local_path = local_data.split(local_data.rstrip('/').split('/')[-1])[0]
        cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)

        cloud_objects = subprocess.run(['rclone', 'lsf', cloud_path], capture_output = True, encoding = 'utf-8').stdout.split()

        if relative_name + '/' in cloud_objects: #directory
            output = subprocess.run(['rclone', 'copy', cloud_path + relative_name, local_path + relative_name], capture_output = True, encoding = 'utf-8')
        elif relative_name in cloud_objects: #file
            output = subprocess.run(['rclone', 'copy', cloud_path + relative_name, local_path], capture_output = True, encoding = 'utf-8')
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
            output = subprocess.run(['tar', '-xvf', local_path + relative_name, '-C', local_path], capture_output = True, encoding = 'utf-8')
            output = subprocess.run(['rm', '-f', local_path + relative_name], capture_output = True, encoding = 'utf-8')

        if tarred_subdirs:
            for d in [x for x in os.listdir(local_data) if '.tar' in x]:
                output = subprocess.run(['tar', '-xvf', local_data + d, '-C', local_data, '--strip-components', '1'], capture_output = True, encoding = 'utf-8')
                os.remove(local_data + d)

    def uploadData(self, local_data, tarred = False):

        relative_name = local_data.rstrip('/').split('/')[-1]
        local_path = local_data.split(relative_name)[0]
        cloud_path = local_path.replace(self.localMasterDir, self.cloudMasterDir)

        if tarred:
            output = subprocess.run(['tar', '-cvf', local_path + relative_name + '.tar', '-C', local_path, relative_name], capture_output = True, encoding = 'utf-8')
            if output.returncode != 0:
                print(output.stderr)
                raise Exception('Error in tarring ' + local_data)
            relative_name += '.tar'

        if os.path.isdir(local_path + relative_name):
            output = subprocess.run(['rclone', 'copy', local_path + relative_name, cloud_path + relative_name], capture_output = True, encoding = 'utf-8')
            #subprocess.run(['rclone', 'check', local_path + relative_name, cloud_path + relative_name], check = True) #Troubleshooting directory will have depth data in it when you upload the cluster data

        elif os.path.isfile(local_path + relative_name):
            print(['rclone', 'copy', local_path + relative_name, cloud_path])
            output = subprocess.run(['rclone', 'copy', local_path + relative_name, cloud_path], capture_output = True, encoding = 'utf-8')
            output = subprocess.run(['rclone', 'check', local_path + relative_name, cloud_path], check = True, capture_output = True, encoding = 'utf-8')
        else:
            raise Exception(local_data + ' does not exist for upload')

        if output.returncode != 0:
            pdb.set_trace()
            raise Exception('Error in uploading file: ' + output.stderr)

    def uploadAndMerge(self, local_data, master_file, tarred = False, ID = False):
        if os.path.isfile(local_data):
            #We are merging two crv files
            self.downloadData(master_file)
            import pandas as pd
            if ID:
                old_dt = pd.read_csv(master_file, index_col = ID)
                new_dt = pd.read_csv(local_data, index_col = ID)
                old_dt = old_dt.append(new_dt)
                old_dt.index.name = ID
            else:
                old_dt = pd.read_csv(master_file)
                new_dt = pd.read_csv(local_data)
                old_dt = old_dt.append(new_dt)
            
            old_dt.to_csv(master_file, sep = ',')
            self.uploadData(master_file)
        else:
            #We are merging two tarred directories
            try:        
                self.downloadData(master_file, tarred = True)
            except FileNotFoundError:
                self.createDirectory(master_file)
            for nfile in os.listdir(local_data):
                subprocess.run(['mv', local_data + nfile, master_file])
            self.uploadData(master_file, tarred = True)
