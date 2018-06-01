import os
import pandas as pd
import cv2
import numpy as np
'''
   1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Integer (0,1,2) indicating the level of truncation.
                     Note that this is in contrast to the object detection
                     benchmark where truncation is a float in [0,1].
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''
class KITTIDataReader:
    def __init__(self, image_folder, detection_file_name):
        datatype = {0: int, 1: int, 2: str, 3: int, 4: int, 5: float,
                    6: float, 7: float, 8: float, 9: float,
                    10: int, 11: int, 12: int, 13: int, 14: int,
                    15: int, 16: int, 17: float}
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.image_format = os.path.join(self.image_folder, '{0:06d}.png')
        self.detection = pd.read_csv(self.detection_file_name,  sep=' ', header=None, dtype=datatype)

        self.detection = self.detection.iloc[:, 0:18]
        select_type_row = [t in ('Van', 'Car', 'Pedestrian', 'Tram', 'Cyclist', 'Truck') for t in self.detection[2]] #Pedestrian
        self.detection = self.detection[select_type_row]
        select_score_row = [t >= 0 for t in self.detection[17]]
        self.detection = self.detection[select_score_row]
        select_occluded_row = [t in [0, 1] for t in self.detection[4]]
        self.detection = self.detection[select_occluded_row]

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
        return (self.get_image_by_index(item),
                self.get_detection_by_index(item))


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

