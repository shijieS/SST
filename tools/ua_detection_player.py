import cv2
from data.ua_detection_data_reader import UADetectionDataReader
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='MITTI tracking player')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--root', default='/media/jianliu/ssm/dataset/dataset/UA-DETRAC', help='UA ROOT')
parser.add_argument('--detection_folder_name', default='EB', help='ground truth of detection folder name')
parser.add_argument('--ignore_folder_name', default='igrs', help='ignore folder name')
parser.add_argument('--use_ignore', default=True, help='use ignore file or not')
parser.add_argument('--show_ignore_rect', default=True, help='whether to show ignored rectangles or not')
parser.add_argument('--image_folder_name', default='Insight-MVT_Annotation_Train', help='The image folder name')
parser.add_argument('--detection_threshold', default=0.1, type=float, help='The minimum threshold of detection')

args = parser.parse_args()


def play():
    # 1. show the information
    print('=====================')
    print('play the the tracking result now.')
    print('=====================')

    # 2. read all folders in image_root
    image_root = os.path.join(args.root, args.image_folder_name)
    all_image_folders = sorted(
        [os.path.join(image_root, d) for d in os.listdir(image_root)]
    )

    # 3. read all files in detection_root
    detection_root = os.path.join(args.root, args.detection_folder_name)
    all_detection_files = sorted(
        [os.path.join(detection_root, f) for f in os.listdir(detection_root) if 'MVI_' in f]
    )

    # 3.1 read all ignore files in igrs_root
    ignore_root = os.path.join(args.root, args.ignore_folder_name)
    all_ignore_files = sorted(
        [os.path.join(ignore_root, f) for f in os.listdir(ignore_root)]
    )

    ignore_file_base_name = [os.path.basename(f)[:-8] for f in all_ignore_files]
    detection_file_base_name = [os.path.basename(f)[:9] for f in all_detection_files]
    # 4. for each item in image_root create a reader
    for image_folder in all_image_folders:

        i = ignore_file_base_name.index(os.path.basename(image_folder))
        ignore_file = all_ignore_files[i]

        j = detection_file_base_name.index(os.path.basename(image_folder))
        detection_file = all_detection_files[j]

        if not os.path.exists(image_folder) or not os.path.exists(detection_file) or not os.path.exists(ignore_file):
            continue
        print('processing ', image_folder, '>>>>>>>>>>>>>>>>>>')
        # 4.1 create a reader
        reader = UADetectionDataReader(image_folder, detection_file, ignore_file if args.use_ignore else None, args.detection_threshold)
        # 4.2 read all the item from the reader
        for i in range(len(reader)):
            # 4.2.1 read the detection
            image, data = reader[i]
            if data is None or image is None:
                continue

            # 4.2.2 show the result
            dets = data[:, 2:6].astype(int)

            for det in dets:
                cv2.rectangle(image, (det[0], det[1]), (det[2], det[3]), tuple((np.random.rand(3) * 255).astype(int).tolist()), 2, )

            if args.show_ignore_rect:
                if not reader.ignore is None:
                    for r in reader.ignore:
                        image = cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 255, 255), 2)

            cv2.imshow('ua tracking player', image)
            cv2.waitKey(25)


if __name__ == '__main__':
    play()


