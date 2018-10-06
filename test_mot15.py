from tracker import SSTTracker, TrackerConfig, Track
# from sst_tracker import TrackSet as SSTTracker
import cv2
from data.mot_data_reader import MOTDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os

parser = argparse.ArgumentParser(description='Single Shot Tracker Test')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--mot_root', default=config['mot_root'], help='MOT ROOT')
parser.add_argument('--type', default=config['type'], help='train/test')
parser.add_argument('--show_image', default=False, help='show image if true, or hidden')
parser.add_argument('--save_video', default=True, help='save video if true')
parser.add_argument('--log_folder', default=config['log_folder'], help='video saving or result saving folder')
parser.add_argument('--mot_version', default=15, help='mot version')

args = parser.parse_args()


def test(choice=None):
    video_name_list = config['video_name_list']

    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    save_folder = ''
    choice_str = ''
    if not choice is None:
        choice_str = TrackerConfig.get_configure_str(choice)
        save_folder = os.path.join(args.log_folder, choice_str)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    save_file_name_format = os.path.join(save_folder, '{}.txt')
    save_video_name_format = os.path.join(save_folder, '{}.avi')
    timer = Timer()

    for video_name in video_name_list:
        if video_name == 'AVG-TownCentre':
            TrackerConfig.set_configure((4, 0, 4, 4, 5, 4))
        else:
            TrackerConfig.set_configure(choice)


        mot_root = os.path.join(config['mot_root'], config['dataset_type'])
        mot_root = os.path.join(mot_root, video_name)
        image_folder = os.path.join(mot_root, 'img1')
        detection_file_name = os.path.join(mot_root, 'det/det.txt')
        save_video_name = save_video_name_format.format(video_name)
        save_file_name = save_file_name_format.format(video_name)
        reader = MOTDataReader(image_folder=image_folder,
                               detection_file_name=detection_file_name,
                               min_confidence=None)
        tracker = SSTTracker()

        result = list()
        result_str = save_file_name

        force_init = True
        for i, item in enumerate(reader):
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

            if i == 1 and args.save_video:
                vw = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))
            det = det.astype(float)
            det[:, [2, 4]] /= float(w)
            det[:, [3, 5]] /= float(h)
            timer.tic()
            image_org = tracker.update(img, det[:, 2:6], args.show_image, i, force_init)
            force_init = False
            timer.toc()
            print('{}:{}, {}, {}\r'.format(os.path.basename(save_file_name), i, int(i * 100 / len(reader)), choice_str))
            if args.show_image and not image_org is None:
                cv2.imshow('res', image_org)
                cv2.waitKey(1)

            if args.save_video and not image_org is None:
                try:
                    vw.write(image_org)
                except:
                    pass

            # save result
            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(tracker.frame_index - 1, tracker.recorder)
                    result.append(
                        [i+1] + [t.id+1] + [b[0] * w, b[1] * h, b[2] * w, b[3] * h] + [-1, -1, -1, -1]
                    )
        # save data
        np.savetxt(save_file_name, np.array(result).astype(int), fmt='%i')
        print(result_str)

    print(timer.total_time)
    print(timer.average_time)

if __name__ == '__main__':
    all_choices = TrackerConfig.get_choices_age_node()
    iteration = 3
    # test()

    i = 0
    for age in range(1):
        for node in range(1):
            c = (0, 0, 4, 4, 5, 4)
            choice_str = TrackerConfig.get_configure_str(c)
            TrackerConfig.set_configure(c)
            print('=============================={}.{}=============================='.format(i, choice_str))
            test(c)
            i += 1

    # for i in range(10):
    #     c = all_choices[-i]
    #
    #     choice_str = TrackerConfig.get_configure_str(c)
    #     TrackerConfig.set_configure(c)
    #     print('=============================={}.{}=============================='.format(i, choice_str))
    #     test(c)
