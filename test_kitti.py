# from tracker import SSTTracker
from sst_tracker import TrackSet as SSTTracker
import cv2
from data.kitti_detection_data_reader import KITTIDataReader
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os

parser = argparse.ArgumentParser(description='Single Shot Joint Tracker Test')
parser.add_argument('--version', default='v1', help='current version')
parser.add_argument('--kitti_image_root', default=config['kitti_image_root'], help='MOT ROOT')
parser.add_argument('--kitti_detection_root', default=config['kitti_detection_root'], help='MOT ROOT')
parser.add_argument('--type', default=config['dataset_type'], help='training/testing')
parser.add_argument('--show_image', default=True, help='show image if true, or hidden')
parser.add_argument('--save_video', default=False, help='save video if true')

args = parser.parse_args()

def test():
    image_root = os.path.join(args.kitti_image_root, args.type, 'image_02')
    all_image_folders = sorted(
        [os.path.join(image_root, d) for d in os.listdir(image_root)]
    )
    detection_root = os.path.join(args.kitti_detection_root, args.type, 'det_02')
    all_detection_files = sorted(
        [os.path.join(detection_root, f) for f in os.listdir(detection_root)]
    )
    saved_file_name_format = 'KITTI'+'-{0:04}.txt'
    save_video_name_format = 'KITTI'+'-{0:04}.avi'

    timer = Timer()
    i = 0
    for image_folder, detection_file_name in zip(all_image_folders, all_detection_files):
        save_file_name = saved_file_name_format.format(i)
        save_video_name = save_video_name_format.format(i)

        print('start processing '+save_file_name)
        tracker = SSTTracker()
        reader = KITTIDataReader(image_folder = image_folder,
                      detection_file_name =detection_file_name)
        i = 0
        result = list()
        result_str = save_file_name


        for item in reader:
            i += 1
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

            det[:, [8,9]] = det[:, [8,9]] - det[:, [6,7]]
            det[:, [6,8]] /= float(w)
            det[:, [7,9]] /= float(h)
            timer.tic()
            image_org = tracker.update(img, det[:, 6:10], args.show_image)
            timer.toc()
            print('{}:{}, {}\r'.format(save_file_name, i, int(i*100/len(reader))))
            if args.show_image and not image_org is None and image_org.shape[0]>0:
                cv2.imshow('res', image_org)
                cv2.waitKey(0)

            if args.save_video and not image_org is None  and image_org.shape[0]>0:
                vw.write(image_org)

            # save result
            for t in tracker.tracks:
                continue
                n = t.nodes[-1]
                if t.age == 1:
                    b = n.box
                    result.append(
                        [i] + [t.id] + [b[0]*w, b[1]*h, b[2]*w, b[3]*h] + [-1, -1, -1, -1]
                    )
        # save data
        np.savetxt(save_file_name, np.array(result).astype(int), fmt='%i')
        print(result_str)

    print(timer.total_time)
    print(timer.average_time)

if __name__ == '__main__':
    test()
