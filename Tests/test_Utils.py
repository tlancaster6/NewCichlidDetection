import unittest
import Utils.ConfigurationLoader as cl
from Utils.FileManager import FileManager
from Utils.DataLoader import load_boxed_annotation_data, BoxedImageLoader
from Utils.YOLOConfig import partition_dataset
import os


class ConfigurationTest(unittest.TestCase):
    def test_load_model_config(self):
        model_config = cl.load_model_config()
        self.assertNotEqual(model_config, None)
        print(model_config)

    def test_load_wrong_model_config(self):
        env = cl.Environment(model='terrible_model')
        self.assertRaises(KeyError, lambda: cl.load_model_config(env=env))


class FileManagerTest(unittest.TestCase):
    def test_download_data_file(self):
        environment = cl.load_environment()
        fm = FileManager(environment)
        fm.download_data(environment.annotated_data_list)
        self.assertTrue(os.path.exists(fm.map_relative_path_to_local(environment.annotated_data_list)))

    def test_download_data_folder(self):
        environment = cl.load_environment()
        fm = FileManager(environment)
        fm.download_data(environment.annotated_data_folder)
        self.assertTrue(os.path.exists(fm.map_relative_path_to_local(environment.annotated_data_folder)))


class DataLoaderTest(unittest.TestCase):
    def test_load_data(self):
        environment = cl.load_environment()
        load_boxed_annotation_data(environment)

    def test_yolo(self):
        env = cl.load_environment()
        partition_dataset(env)

    def test_dataset(self):
        env = cl.load_environment()
        data = load_boxed_annotation_data(env, download=False)
        dataset = BoxedImageLoader(data)


if __name__ == '__main__':
    unittest.main()
