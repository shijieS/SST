from tracker import SSTTracker, TrackerConfig, Track
import cv2
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os
from tqdm import trange
from joblib import Parallel, delayed
import pandas as pd
from evaluate_mot import \
    read_gt_mot17, \
    get_summary_mot17, \
    read_test_mot17_single_file, \
    mot17_post_processing


parser = argparse.ArgumentParser(description='Single Shot Tracker Test')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--mot_root', default=config['mot_root'], help='MOT ROOT')
parser.add_argument('--type', default=config['type'], help='train/test')
parser.add_argument('--select_index', default=-1, type= int, help='selected video indexes')
parser.add_argument('--show_image', default=False, help='show image if true, or hidden')
parser.add_argument('--save_video', default=False, help='save video if true')
parser.add_argument('--log_folder', default=config['log_folder'], help='video saving or result saving folder')
parser.add_argument('--mot_version', default=17, help='mot version')

args = parser.parse_args()


best_config = {
    'SDP': {
        2:  [0.9, 4, 2, 3, 1, 3, 3, 3],
        4:  [0.0, 2, 4, 1, 1, 4, 3, 3],
        5:  [0.6, 2, 0, 1, 0, 4, 4, 2],
        9:  [0.9, 2, 3, 1, 0, 4, 3, 3],
        10: [0.6, 4, 0, 3, 1, 4, 3, 2],
        11: [0.6, 2, 3, 0, 4, 4, 4, 4],
        13: [0.9, 4, 2, 4, 0, 3, 2, 2]
    },
    'DPM': {
        2:  [0.0, 1, 0, 0, 0, 4, 4, 4],
        4:  [0.0, 1, 1, 2, 1, 4, 4, 3],
        5:  [0.0, 2, 3, 4, 1, 2, 3, 4],
        9:  [0.0, 1, 1, 4, 3, 3, 2, 1],
        10: [0.0, 4, 4, 0, 1, 2, 3, 2],
        11: [0.3, 4, 3, 1, 0, 4, 4, 3],
        13: [0.0, 3, 1, 1, 0, 3, 3, 4]
    },
    'FRCNN': {
        2:  [0.0, 2, 2, 1, 1, 3, 4, 4],
        4:  [0.0, 4, 1, 2, 2, 3, 4, 3],
        5:  [0.0, 2, 2, 0, 2, 3, 2, 0],
        9:  [0.0, 1, 1, 4, 3, 3, 2, 1],
        10: [0.9, 2, 0, 3, 4, 4, 4, 2],
        11: [0.0, 2, 0, 0, 2, 3, 3, 2],
        13: [0.9, 3, 3, 0, 0, 4, 4, 1]

    }
}


def evaluate(choice,
             dataset_type=args.type,
             dataset_indexes=None,
             detector_names=None,
             min_confidence=config['min_confidence'],
             gt=None):

    selected_file_names = get_selected_evaluate_names(
        dataset_type,
        dataset_indexes,
        detector_names
    )

    # get image folders
    dataset_root = os.path.join(args.mot_root, dataset_type)

    # get save folders
    choice_str = TrackerConfig.get_configure_str(choice)
    save_folder = os.path.join(args.log_folder, choice_str)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # setting tracker
    TrackerConfig.set_configure(choice)

    # creat summary lists if gt exists
    if gt:
        summaries = {}

    # start evaluate each videos
    timer = Timer()
    for base_name in selected_file_names:
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

            det[:, [2,4]] /= float(w)
            det[:, [3,5]] /= float(h)
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

        # use motmetric to evaluate the saved text
        if gt:

            ts = read_test_mot17_single_file(mot17_post_processing(save_txt_file))
            summary = get_summary_mot17(gt, ts)
            summaries[base_name] = summary
    return summaries



def get_selected_evaluate_names(
        dataset_type,
        dataset_indexes=None,
        detector_names=None
):
    """
    Get the selected video base name. If *dataset_indexes* is `None`, we use all the `train indexes` or `test indexes`. It is same for `detector_names`

    Args:
        dataset_type: ['train', 'test']
        dataset_indexes: The selected dataset indexes list, default is None.
        detector_names: The selected detector names, default is None.

    Returns: The list of selected file base name.

    """

    if dataset_type == 'train':
        if not dataset_indexes:
            dataset_indexes = [2, 4, 5, 9, 10, 11, 13]
        if not detector_names:
            detector_names = {'DPM', 'FRCNN', 'SDP'}

    if dataset_type == 'test':
        if not dataset_indexes:
            dataset_indexes = [1, 3, 6, 7, 8, 12, 14]
        if not detector_names:
            detector_names = {'FRCNN', 'DPM', 'SDP'}

    selected_file_names = [
        'MOT17-{:02}-{}'.format(i, j)
        for j in detector_names
        for i in dataset_indexes
    ]
    return selected_file_names


def run_test(search_num=100,
             dataset_indexes=[2],
             detector_names={'DPM'},
             save_name='Result-DPM-02.csv',
             gt=None):
    dataset_indexes = dataset_indexes
    detector_names = detector_names

    all_choices = TrackerConfig.get_all_choices()
    select_index = list(range(len(all_choices)))
    np.random.shuffle(select_index)
    select_index = select_index[:search_num]

    # all_choices = [[1, 1, 2, 1, 4, 4, 3]]
    # select_index = [0]

    if not gt:
        gt = read_gt_mot17('./evaluate_mot/ground_truth/train')
    confidences = [0.3*i for i in range(4)]
    # confidences = [0.0]
    result = {}
    all_choices = [(co, all_choices[s]) for co in confidences for s in select_index]
    for i_, (con, choice) in zip(trange(len(all_choices)), all_choices):
        choice_str = '{}-{}-{}-{}-{}-{}-{}-{}'.format(con, *tuple(choice))
        summaries = evaluate(
            choice=choice,
            dataset_type='train',
            dataset_indexes=dataset_indexes,
            detector_names=detector_names,
            min_confidence=con,
            gt=gt
        )
        result[choice_str] = summaries

    result_list = []
    for c in result.keys():
        for base_name in result[c].keys():
            result_list += [
                result[c][base_name].iloc[-1:, :].rename(index={'OVERALL': str(c)})
            ]
    result_merge = pd.concat(result_list)
    result_merge.to_csv(
        os.path.join(args.log_folder, save_name)
    )


def select_best_params():
    dataset_indexes = [2, 4, 5, 9, 10, 11, 13]
    detector_names = {'DPM', 'FRCNN', 'SDP'}
    gt = read_gt_mot17('./evaluate_mot/ground_truth/train')

    all_scene = [([i], {j}) for i in dataset_indexes for j in detector_names]

    if args.select_index > -1:
        all_scene = [all_scene[args.select_index]]
    print(all_scene)

    # all_scene = [([4], {'DPM'})]


    for _i, (indexes, names) in zip(trange(len(all_scene)), all_scene):
        save_name="{}-{}.csv".format(list(names)[0], indexes[0])
        run_test(search_num=30,
                 dataset_indexes=indexes,
                 detector_names=names,
                 save_name=save_name,
                 gt=gt)


if __name__ == "__main__":
    select_best_params()
