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
parser.add_argument('--show_image', default=True, help='show image if true, or hidden')
parser.add_argument('--save_video', default=True, help='save video if true')
parser.add_argument('--log_folder', default=config['log_folder'], help='video saving or result saving folder')
parser.add_argument('--mot_version', default=17, help='mot version')

args = parser.parse_args()


def test(choice=None):
    if args.type == 'train':
        dataset_index = [2, 4, 5, 9, 10, 11, 13]
        dataset_detection_type = {'-DPM', '-FRCNN', '-SDP'}

    if args.type == 'test':
        dataset_index = [1, 3, 6, 7, 8, 12, 14]
        dataset_detection_type = {'-FRCNN', '-SDP', '-DPM'}

    dataset_image_folder_format = os.path.join(args.mot_root, args.type+'/MOT'+str(args.mot_version)+'-{:02}{}/img1')
    detection_file_name_format=os.path.join(args.mot_root, args.type+'/MOT'+str(args.mot_version)+'-{:02}{}/det/det.txt')

    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    save_folder = ''
    choice_str = ''
    if not choice is None:
        choice_str = TrackerConfig.get_configure_str(choice)
        save_folder = os.path.join(args.log_folder, choice_str)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        # else:
        #     return

    saved_file_name_format = os.path.join(save_folder, 'MOT'+str(args.mot_version)+'-{:02}{}.txt')
    save_video_name_format = os.path.join(save_folder, 'MOT'+str(args.mot_version)+'-{:02}{}.avi')

    f = lambda format_str: [format_str.format(index, type) for type in dataset_detection_type for index in dataset_index]

    timer = Timer()
    for image_folder, detection_file_name, saved_file_name, save_video_name in zip(f(dataset_image_folder_format), f(detection_file_name_format), f(saved_file_name_format), f(save_video_name_format)):
        print('start processing '+saved_file_name)
        tracker = SSTTracker()
        reader = MOTDataReader(image_folder = image_folder,
                      detection_file_name =detection_file_name,
                               min_confidence=0.0)
        result = list()
        result_str = saved_file_name
        first_run = True
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


            if first_run and args.save_video:
                vw = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h))
                first_run = False

            det[:, [2,4]] /= float(w)
            det[:, [3,5]] /= float(h)
            timer.tic()
            image_org = tracker.update(img, det[:, 2:6], args.show_image, i)
            timer.toc()
            print('{}:{}, {}, {}\r'.format(os.path.basename(saved_file_name), i, int(i*100/len(reader)), choice_str))
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
                        [i] + [t.id] + [b[0]*w, b[1]*h, b[2]*w, b[3]*h] + [-1, -1, -1, -1]
                    )
        # save data
        np.savetxt(saved_file_name, np.array(result).astype(int), fmt='%i')
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
            c = (0, 0, 4, 0, 3, 3)
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
