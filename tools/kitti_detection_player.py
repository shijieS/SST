from tracker import SSTTracker
import cv2
from data.kitti_detection_data_reader import  KITTIDataReader
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os

parser = argparse.ArgumentParser(description='MITTI tracking player')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--kitti_image_root', default='/home/ssm/ssj/dataset/KITTI/tracking/image_2', help='MOT ROOT')
parser.add_argument('--kitti_detection_root', default='/home/ssm/ssj/dataset/KITTI/tracking/det_2_lsvm', help='MOT ROOT')
parser.add_argument('--type', default='training', help='only training allowed')

args = parser.parse_args()

def show_rectangle(image, box):
    color = tuple((np.random.rand(3) * 255).astype(int).tolist())
    image = cv2.rectangle(image, tuple(box[:2]), tuple(box[2:4]), color, 2)
    return image

def play():
    # 1. show the information
    print('=====================')
    print('play the rectangles.')
    print('=====================')

    # 2. read all folders in image_root
    image_root = os.path.join(args.kitti_image_root, args.type, 'image_02')
    all_image_folders = sorted(
        [os.path.join(image_root, d) for d in os.listdir(image_root)]
    )

    # 3. read all files in detection_root
    detection_root = os.path.join(args.kitti_detection_root, args.type, 'det_02')
    all_detection_files = sorted(
        [os.path.join(detection_root, f) for f in os.listdir(detection_root)]
    )

    # 4. for each item in image_root create a reader
    for image_folder, detection_file in zip(all_image_folders, all_detection_files):
        if not os.path.exists(image_folder) or not os.path.exists(detection_file):
            continue

        image_file_format = os.path.join(image_folder, '{0:04}.png')

        # 4.1 create a reader
        reader = KITTIDataReader(image_folder, detection_file)
        # 4.2 read all the item from the reader
        for i in range(len(reader)):
            # 4.2.1 read the detection
            image, data = reader[i]
            if data is None or image is None:
                continue

            # 4.2.2 show rectangles
            dets = data[:, 6:10].astype(int)

            for d in dets:
                image = show_rectangle(image, d)

            cv2.imshow('kitti tracking player', image)
            cv2.waitKey(0)

if __name__ == '__main__':
    play()


