from tracker import SSTTracker
import cv2
from data.kitti_tracking_data_reader import  KITTITrackingDataReader
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os

parser = argparse.ArgumentParser(description='MITTI tracking player')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--kitti_image_root', default='/home/ssm/ssj/dataset/KITTI/tracking/image_2', help='MOT ROOT')
parser.add_argument('--kitti_detection_root', default='/home/ssm/ssj/dataset/KITTI/tracking/tracking_label_2/', help='KITTI ROOT')
parser.add_argument('--type', default='training', help='only training allowed')
parser.add_argument('--max_boxes', default=50, help='an integer to indicate the max boxes of the track')
parser.add_argument('--max_age', default= 50, help='the minize visibility')

args = parser.parse_args()

def play():
    # 1. show the information
    print('=====================')
    print('play the the tracking result now.')
    print('=====================')

    # 2. read all folders in image_root
    image_root = os.path.join(args.kitti_image_root, args.type, 'image_02')
    all_image_folders = sorted(
        [os.path.join(image_root, d) for d in os.listdir(image_root)]
    )

    # 3. read all files in detection_root
    detection_root = os.path.join(args.kitti_detection_root, args.type, 'label_02')
    all_detection_files = sorted(
        [os.path.join(detection_root, f) for f in os.listdir(detection_root)]
    )

    # 4. for each item in image_root create a reader
    for image_folder, detection_file in zip(all_image_folders, all_detection_files):
        if not os.path.exists(image_folder) or not os.path.exists(detection_file):
            continue

        image_file_format = os.path.join(image_folder, '{0:04}.png')

        # 4.1 create a reader
        reader = KITTITrackingDataReader(image_folder, detection_file)
        trackers = Tracks()
        # 4.2 read all the item from the reader
        for i in range(len(reader)):
            # 4.2.1 read the detection
            image, data = reader[i]
            if data is None or image is None:
                continue

            # 4.2.2 update tracks
            dets = data[:, 6:10].astype(int)
            ids = data[:, 1]

            for id, d in zip(ids, dets):
                if id == -1:
                    continue
                trackers.add_box(d, id)

            trackers.one_frame_pass()

            # 4.2.3 show the trackers
            image = trackers.show(image)

            cv2.imshow('kitti tracking player', image)
            cv2.waitKey(0)



class Track:
    def __init__(self, id=-1):
        self.boxes = list()
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())
        self.id = id
        self.age = 0

    def add_box(self, b):
        if len(self.boxes) == args.max_boxes:
            del self.boxes[0]

        self.boxes.append(b)
        self.age = 0

    def one_frame_pass(self):
        self.age += 1

    def show(self, image):
        if len(self.boxes) == 0:
            return image

        for b_next, b_pre in zip(self.boxes[1:], self.boxes):
            p_pre = tuple(((b_pre[:2] + b_pre[2:])/2).astype(int))
            p_next = tuple(((b_next[:2] + b_next[2:]) / 2).astype(int))
            image = cv2.line(image, tuple(p_pre), tuple(p_next), self.color, 2)

        b = self.boxes[-1]
        image = cv2.rectangle(image, tuple(b[:2]), tuple(b[2:4]), self.color, 2)

        return image


class Tracks:
    def __init__(self):
        self.tracks = list()
    def show(self, image):
        for t in self.tracks:
            if t.age == 1:
                image = t.show(image)
        return image

    def add_box(self, box, id):
        box_add = False
        for t in self.tracks:
            if t.id==id:
                t.add_box(box)
                box_add = True
                break
        if not box_add:
            t = Track(id)
            t.add_box(box)
            self.tracks.append(t)

    def one_frame_pass(self):
        del_indexes = list()
        for i, t in enumerate(self.tracks):
            t.one_frame_pass()
            if t.age > args.max_age:
                del_indexes.append(i)

        for i in reversed(del_indexes):
            del self.tracks[i]



if __name__ == '__main__':
    play()


