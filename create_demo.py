from tracker import SSTTracker, TrackerConfig, Track
# from sst_tracker import TrackSet as SSTTracker
import cv2
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os

parser = argparse.ArgumentParser(description='Demo images creating')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--mot_root', default=config['mot_root'], help='MOT ROOT')
parser.add_argument('--type', default=config['type'], help='train/test')
parser.add_argument('--show_image', default=True, help='show image if true, or hidden')
parser.add_argument('--log_folder', default=config['log_folder'], help='video saving or result saving folder')
parser.add_argument('--mot_version', default=17, help='mot version')

args = parser.parse_args()


selected_frames = [100+5*i for i in range(6)]
image_folder = os.path.join(args.mot_root, 'train/MOT17-09-FRCNN/img1')
detection_file_name = os.path.join(args.mot_root, 'train/MOT17-09-FRCNN/det/det.txt')

def create(c):
    if not os.path.exists(image_folder) or not os.path.exists(detection_file_name):
        raise FileNotFoundError('cannot find the image folder and the detection file')

    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    tracker = SSTTracker()
    reader = MOTDataReader(image_folder=image_folder,
                  detection_file_name=detection_file_name,
                           min_confidence=0.0)

    select_squences = [402, 404, 410, 422]

    frame_index = 0
    for i, item in enumerate(reader):
        if i not in select_squences:
            continue

        if i > len(reader):
            break

        if item is None:
            continue

        img = item[0]
        det = item[1]

        if img is None or det is None or len(det) == 0:
            continue

        if len(det) > config['max_object']:
            det = det[:config['max_object'], :]

        h, w, _ = img.shape

        det[:, [2, 4]] /= float(w)
        det[:, [3, 5]] /= float(h)

        image_org = tracker.update(img, det[:, 2:6], args.show_image, frame_index)
        frame_index += 1

        if args.show_image and not image_org is None:
            # det[:, [2, 4]] *= float(w)
            # det[:, [3, 5]] *= float(h)
            # boxes = det[:, 2:6].astype(int)
            # for bid, b in enumerate(boxes):
            #     image_org = cv2.putText(image_org, '{}'.format(bid), tuple(b[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                             (0, 0, 0), 2)
            cv2.imshow('res', image_org)
            cv2.imwrite(os.path.join(args.log_folder, '{0:06}.jpg'.format(i)), image_org)
            # cv2.waitKey(0)
            print('frame: {}'.format(i))



if __name__ == '__main__':
    c = (0, 0, 4, -1, 5, 4)
    TrackerConfig.set_configure(c)
    create(c)
