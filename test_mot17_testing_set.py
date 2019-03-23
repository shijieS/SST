from tracker import SSTTracker, TrackerConfig, Track
import cv2
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os
from tqdm import trange
import pandas as pd
from evaluate_mot import \
    read_gt_mot17, \
    get_summary_mot17, \
    read_test_mot17_file_list, \
    mot17_post_processing


parser = argparse.ArgumentParser(description='Single Shot Tracker Test')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--mot_root', default=config['mot_root'], help='MOT ROOT')
parser.add_argument('--type', default='train', help='train/test')
parser.add_argument('--show_image', default=False, help='show image if true, or hidden')
parser.add_argument('--save_video', default=True, help='save video if true')
parser.add_argument('--log_folder', default=config['log_folder'], help='video saving or result saving folder')
parser.add_argument('--mot_version', default=17, help='mot version')

args = parser.parse_args()


best_config = {
    'SDP': {
        1:  [0.9, 4, 2, 3, 1, 3, 3, 3],
        3:  [0.0, 2, 4, 1, 1, 4, 3, 3],
        6:  [0.6, 2, 0, 1, 0, 4, 4, 2],
        7:  [0.9, 2, 3, 1, 0, 4, 3, 3],
        8: [0.6, 4, 0, 3, 1, 4, 3, 2],
        12: [0.6, 2, 3, 0, 4, 4, 4, 4],
        14: [0.9, 4, 2, 4, 0, 3, 2, 2]
    },
    'DPM': {
        1:  [-0.5, 1, 0, 0, 0, 4, 4, 4],
        3:  [-0.5, 1, 1, 2, 1, 4, 4, 3],
        6:  [-0.5, 2, 3, 4, 1, 2, 3, 4],
        7:  [-0.5, 1, 1, 4, 3, 3, 2, 1],
        8:  [-0.5, 4, 4, 0, 1, 2, 3, 2],
        12: [-0.5, 4, 3, 1, 0, 4, 4, 3],
        14: [-0.5, 3, 1, 1, 0, 3, 3, 4]
    },
    'FRCNN': {
        1:  [0.0, 2, 2, 1, 1, 3, 4, 4],
        3:  [0.0, 4, 1, 2, 2, 3, 4, 3],
        6:  [0.0, 2, 2, 0, 2, 3, 2, 0],
        7:  [0.0, 1, 1, 4, 3, 3, 2, 1],
        8:  [0.9, 2, 0, 3, 4, 4, 4, 2],
        12: [0.0, 2, 0, 0, 2, 3, 3, 2],
        14: [0.9, 3, 3, 0, 0, 4, 4, 1]

    }
}


def evaluate(dataset_type=args.type,
             selected_file_names=None,
             selected_config=None):

    # get image folders
    dataset_root = os.path.join(args.mot_root, dataset_type)

    # creat summary lists if gt exists

    saved_file_list = []

    # start evaluate each videos
    timer = Timer()
    for base_name, choice in zip(selected_file_names, selected_config):

        min_confidence = choice[0]
        choice = choice[1:]

        # get save folders
        choice_str = TrackerConfig.get_configure_str(choice)
        save_folder = os.path.join(args.log_folder, choice_str)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # setting tracker
        TrackerConfig.set_configure(choice)

        # print('======>Start Processing {}'.format(base_name))
        # get image folder & detection file
        image_folder = os.path.join(
            dataset_root,
            '{}/img1'.format(base_name))
        detection_file = os.path.join(
            dataset_root,
            '{}/det/det.txt'.format(base_name))

        # specify the saved text and video path
        save_txt_file = os.path.join(save_folder, '{}.txt'.format(base_name))
        save_video_file = os.path.join(save_folder, '{}.avi'.format(base_name))

        # create tracker
        tracker = SSTTracker()
        reader = MOTDataReader(
            image_folder=image_folder,
            detection_file_name=detection_file,
            min_confidence=min_confidence
        )

        # start tracking each frame in this video
        result = list()
        first_run = True

        for _i, (i, item) in zip(trange(len(reader)), enumerate(reader)):
            if i > len(reader):
                break

            if item is None:
                continue

            img = item[0]
            det = item[1]

            if img is None or det is None or len(det)==0:
                continue

            if len(det) > config['max_object']:
                det = det[:config['max_object'], :]

            h, w, _ = img.shape

            if first_run and args.save_video:
                vw = cv2.VideoWriter(save_video_file, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))
                first_run = False

            det[:, [2, 4]] /= float(w)
            det[:, [3, 5]] /= float(h)
            timer.tic()
            image_org = tracker.update(img, det[:, 2:6], args.show_image, i)
            timer.toc()
            # print('{}:{}, {}, {}\r'.format(os.path.basename(save_txt_file), i, int(i*100/len(reader)), choice_str))
            if args.show_image and not image_org is None:
                cv2.imshow('res', image_org)
                cv2.waitKey(1)

            if args.save_video and not image_org is None:
                vw.write(image_org)

            # save result
            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(tracker.frame_index-1, tracker.recorder)
                    result.append(
                        [i+1] + [t.id+1] + [b[0]*w, b[1]*h, b[2]*w, b[3]*h] + [-1, -1, -1, -1]
                    )
        # save the final result

        np.savetxt(save_txt_file, np.array(result).astype(int), fmt='%i')


if __name__ == "__main__":
    dataset_indexes = [1, 3, 6, 7, 8, 12, 14]
    detector_names = {'FRCNN', 'DPM', 'SDP'}

    selected_file_names = [
        'MOT17-{:02}-{}'.format(i, j)
        for j in detector_names
        for i in dataset_indexes
    ]
    selected_config = [
        best_config[j][i]
        for j in detector_names
        for i in dataset_indexes
    ]

    # gt = read_gt_mot17('./evaluate_mot/ground_truth/train')
    #
    evaluate(dataset_type="test",
             selected_file_names=selected_file_names,
             selected_config=selected_config)



