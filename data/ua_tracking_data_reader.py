import os
import pandas as pd
import cv2
import numpy as np

'''
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
'''


class UADataReader:
    def __init__(self, image_folder, gt_file_name, ignore_file_name):
        datatype = {0: int, 1: int, 2: float, 3: float, 4: float, 5: float}
        datatype_ignore = {0: float, 1: float, 2: float, 3: float}

        self.image_folder = image_folder
        self.gt_file_name = gt_file_name
        self.ignore_file_name = ignore_file_name
        self.image_format = os.path.join(self.image_folder, 'img{0:05d}.jpg')
        self.detection = pd.read_csv(self.gt_file_name, sep=',', header=None, dtype=datatype)

        # read ignore file
        mask = None
        if ignore_file_name is not None and os.stat(self.ignore_file_name).st_size > 0:
            self.ignore = pd.read_csv(self.ignore_file_name, sep=',', header=None, dtype=datatype_ignore)
            self.ignore = self.ignore.values
            ls = self.detection.iloc[:, 2].values
            ts = self.detection.iloc[:, 3].values
            rs = self.detection.iloc[:, 4].values
            bs = self.detection.iloc[:, 5].values

            self.ignore = np.array([[r[0], r[1], r[0]+r[2], r[1]+r[3]] for r in self.ignore])
            for rect in self.ignore:
                l = rect[0]
                t = rect[1]
                r = rect[2]
                b = rect[3]
                res = np.logical_and(
                    np.logical_and(np.logical_and(l < ls, ls < r), np.logical_and(l < rs, rs < r)),
                    np.logical_and(np.logical_and(t < ts, ts < b), np.logical_and(t < bs, bs < b))
                )
                if mask is None:
                    mask = res
                else:
                    mask = np.logical_or(mask, res)
        if mask is not None:
            self.detection = self.detection[np.logical_not(mask)]

        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

    def __len__(self):
        return len(self.detection_group_keys)

    def get_detection_by_index(self, index):
        if index > len(self.detection_group_keys) or self.detection_group_keys.count(index) == 0:
            return None
        return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        if index > len(self.detection_group_keys):
            return None

        return cv2.imread(self.image_format.format(index))

    def __getitem__(self, item):
        return (self.get_image_by_index(item),
                self.get_detection_by_index(item))


class DataTransform:
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
