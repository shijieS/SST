from tracker import SSTTracker, TrackerConfig, Track
import cv2
from data.ua_detection_data_reader import UADetectionDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os

parser = argparse.ArgumentParser(description='Single Shot Tracker Test')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--ua_image_root', default=config['ua_image_root'], help='Image Root')
parser.add_argument('--ua_detection_root', default=config['ua_detection_root'], help='Detection Root')
parser.add_argument('--ua_ignore_root', default=config['ua_ignore_root'], help='Ignore folder Root')
parser.add_argument('--save_folder', default=config['save_folder'], help='save file folder Root')
parser.add_argument('--show_image', default=True, help='show image if true, or hidden')
parser.add_argument('--save_video', default=True, help='save video if true')
parser.add_argument('--use_ignore', default=True, help='use ignore or not')
parser.add_argument('--detection_threshold', default=0.0, help='the threshold of detection')

args = parser.parse_args()


def test(choice=None):
    image_root = args.ua_image_root
    detection_root = args.ua_detection_root
    ignore_root = args.ua_ignore_root
    save_folder = args.save_folder

    if not os.path.exists(image_root) or not os.path.exists(detection_root) or not os.path.exists(ignore_root):
        raise FileNotFoundError('Pls check the file of parameters')

    print('''
    ==============================
    =     Start Reading Files    =
    ==============================
    ''')

    all_image_folders = sorted(
        [os.path.join(image_root, d) for d in os.listdir(image_root)]
    )
    all_detection_files = sorted(
        [os.path.join(detection_root, f) for f in os.listdir(detection_root) if 'MVI_' in f]
    )
    all_ignore_files = sorted(
        [os.path.join(ignore_root, f) for f in os.listdir(ignore_root)]
    )

    ignore_file_base_name = [os.path.basename(f)[:-8] for f in all_ignore_files]
    detection_file_base_name = [os.path.basename(f)[:9] for f in all_detection_files]

    saved_file_name_format = os.path.join(save_folder, '{}.txt')
    saved_video_name_format = os.path.join(save_folder, '{}.avi')

    choice_str = ''
    if not choice is None:
        choice_str = TrackerConfig.get_configure_str(choice)
        if not os.path.exists(choice_str):
            os.mkdir(choice_str)
            save_folder = choice_str + '/'
        else:
            return

    timer = Timer()
    for image_folder in all_image_folders:
        image_folder_base_name = os.path.basename(image_folder)
        i = ignore_file_base_name.index(image_folder_base_name)
        ignore_file = all_ignore_files[i]

        j = detection_file_base_name.index(image_folder_base_name)
        detection_file = all_detection_files[j]

        saved_file_name = saved_file_name_format.format(image_folder_base_name)
        saved_video_name = saved_video_name_format.format(image_folder_base_name)
        vw = None
        if not os.path.exists(image_folder) or not os.path.exists(detection_file) or not os.path.exists(ignore_file):
            continue

        print('processing ', image_folder, '>>>>>>>>>>>>>>>>>>')

        tracker = SSTTracker()
        reader = UADetectionDataReader(image_folder, detection_file, ignore_file if args.use_ignore else None,
                                       args.detection_threshold)

        i = 0
        result = list()
        result_str = saved_file_name

        for item in reader:
            i += 1
            if item is None:
                continue

            img = item[0]
            det = item[1]

            if img is None or det is None or len(det) == 0:
                continue

            if len(det) > config['max_object']:
                det = det[:config['max_object'], :]

            h, w, _ = img.shape
            if vw is None and args.save_video:
                vw = cv2.VideoWriter(saved_video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))

            det[:, [2, 4]] /= float(w)
            det[:, [3, 5]] /= float(h)

            timer.tic()
            image_org = tracker.update(img, det[:, 2:6], args.show_image)
            timer.toc()
            print('{}:{}, {}, {}\r'.format(saved_file_name, i, int(i * 100 / len(reader)), choice_str))

            if args.show_image and not image_org is None:
                cv2.imshow('res', image_org)
                cv2.waitKey(10)

            if args.save_video and not image_org is None:
                vw.write(image_org)

            for t in tracker.tracks:
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.get_box(tracker.frame_index-1, tracker.recorder)
                    result.append(
                        [i] + [t.id] + [b[0]*w, b[1]*h, b[2]*w, b[3]*h] + [-1, -1, -1, -1]
                    )
        # save data
        np.savetxt(saved_file_name, np.array(result).astype(int), fmt='%i')
        print(result_str)

    print(timer.total_time)
    print(timer.average_time)


if __name__ == '__main__':
    all_choices = TrackerConfig.get_all_choices_max_track_node()
    iteration = 3
    test()

    # for i in range(10):
    #     c = all_choices[-i]
    #
    #     choice_str = TrackerConfig.get_configure_str(c)
    #     TrackerConfig.set_configure(c)
    #     print('=============================={}.{}=============================='.format(i, choice_str))
    #     test(c)
