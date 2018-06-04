import os
import pandas as pd
import cv2
import numpy as np


class MOTDataReader:
    def __init__(self, image_folder, detection_file_name):
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, '{0:06d}.jpg')
        self.detection = pd.read_csv(self.detection_file_name, header=None)
        # self.detection = self.detection[self.detection[6] > 0.3]
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

    def __len__(self):
        return len(self.detection_group_keys)

    def get_detection_by_index(self, index):
        if index > len(self.detection_group_keys) or self.detection_group_keys.count(index)==0:
            return None
        return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        if index > len(self.detection_group_keys):
            return None

        return cv2.imread(self.image_format.format(index))

    def __getitem__(self, item):
        return (self.get_image_by_index(item+1),
                self.get_detection_by_index(item+1))


class DataTransform():
    @staticmethod
    def transform(image, detection, size, mean):
        '''
        transform image and detection to the sst input format
        :param image:
        :param detection:
        :param size:
        :param mean:
        :return:
        '''
        h, w, c = image.shape
        image.astype(np.float32)
        detection[[4, 5]] += detection[2, 3]
        image = cv2.resize(image, size)
        image -= mean
        new_h, new_w, new_c = image.shape

