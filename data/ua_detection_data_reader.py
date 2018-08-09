import os
import pandas as pd
import cv2
import numpy as np
'''
   1    frame        Frame within the sequence where the object appearers
   1    detection id detection id of the detected object
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''


class UADetectionDataReader:
    def __init__(self, image_folder, detection_file_name, ignore_file_name, detection_threshold=0.0):
        datatype = {0: int, 1: int, 2: float, 3: float, 4: float, 5: float, 6: float}
        datatype_ignore = {0: float, 1: float, 2: float, 3: float}

        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.ignore_file_name = ignore_file_name
        self.image_format = os.path.join(self.image_folder, 'img{0:05d}.jpg')
        self.detection = pd.read_csv(self.detection_file_name, sep=',', header=None, dtype=datatype)

        # self.detection.iloc[:, 4] += self.detection.iloc[:, 2]
        # self.detection.iloc[:, 5] += self.detection.iloc[:, 3]

        # read ignore file
        mask = None
        self.ignore = None
        if ignore_file_name is not None and os.stat(self.ignore_file_name).st_size > 0:
            self.ignore = pd.read_csv(self.ignore_file_name, sep=',', header=None, dtype=datatype_ignore)
            self.ignore = self.ignore.values
            cx = (2*self.detection.iloc[:, 2].values + self.detection.iloc[:, 4].values)/2.0
            cy = (2*self.detection.iloc[:, 3].values + self.detection.iloc[:, 5].values)/2.0
            self.ignore = np.array([[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in self.ignore])
            for rect in self.ignore:
                l = rect[0]
                t = rect[1]
                r = rect[2]
                b = rect[3]

                res = np.logical_and(
                    np.logical_and(l <= cx, cx <= r),
                    np.logical_and(t <= cy, cy <= b)
                )
                if mask is None:
                    mask = res
                else:
                    mask = np.logical_or(mask, res)
        if mask is not None:
            self.detection = self.detection[np.logical_not(mask)]

        self.detection = self.detection[self.detection.iloc[:, 6] >= detection_threshold]

        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())

        self.length = len(self)
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
        if item >= self.length:
            raise IndexError()
        return (self.get_image_by_index(item+1),
                self.get_detection_by_index(item+1))
