from config.config import config
import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Extract other trackers detection')
parser.add_argument('--tracker_result_path', default=r'D:\ssj\mot\mot15\AP_HWDPL_p', help='MOT ROOT')
parser.add_argument('--save_detection_path', default=r'D:\ssj\mot\mot15\AP_HWDPL_p\det', help='MOT ROOT')

args = parser.parse_args()

if not os.path.exists(args.save_detection_path):
    os.mkdir(args.save_detection_path)


for video_name in config['video_name_list']:
    track_file = os.path.join(args.tracker_result_path, video_name+'.txt')
    detection_file = os.path.join(args.save_detection_path, video_name+'.txt')
    det = pd.read_csv(track_file, header=None, delimiter=',')
    det = det.values
    # det = det[det[:, -4] > 0, :]
    det[:, [1, -1, -2, -3]] = -1
    det[:, -4] = 35
    np.savetxt(detection_file, det, delimiter=',', fmt="%d,%d,%d,%d,%d,%d,%d,%d,%d,%d")

    det_folder = os.path.join(args.save_detection_path, video_name)
    if not os.path.exists(det_folder):
        os.mkdir(det_folder)

    det_folder = os.path.join(det_folder, 'det')
    if not os.path.exists(det_folder):
        os.mkdir(det_folder)

    np.savetxt(os.path.join(det_folder, 'det.txt'), det, delimiter=',', fmt="%d,%d,%d,%d,%d,%d,%d,%d,%d,%d")

    print(video_name)



