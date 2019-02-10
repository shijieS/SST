'''
author: shijie Sun
email: shijieSun@chd.edu.cn
'''

import cv2
import argparse
import os
import numpy as np

print('''
Show your result
Usage: ua_show_result --image_root="ua image root" --result_root="your ua result folder"
''')

parser = argparse.ArgumentParser(description='based on your ua result to draw the image')
parser.add_argument('--image_root', default=r"F:\dataset\dataset\UA-DETRAC\Insight-MVT_Annotation_Test", help='the image folder')
parser.add_argument('--result_root', default=r"F:\dataset\dataset\UA-DETRAC\other-methods-result\Tracker-joint-det-track-id\Tracker-joint-det-track-id\Detector-joint-det-track-id\0.0", help='the destination folders')


args = parser.parse_args()

print('''
start processing ========>>>>>>>>>>
''')

print('''
Reading the results
''')

names = [os.path.splitext(os.path.basename(f))[0][:9] for f in os.listdir(args.result_root) if 'speed' not in os.path.basename(f)]

names = np.unique(names)

colors = {}
for n in names:
    image_folder = os.path.join(args.image_root, n)
    file_H = os.path.join(args.result_root, n+'_H.txt')
    file_W = os.path.join(args.result_root, n+'_W.txt')
    file_LX = os.path.join(args.result_root, n + '_LX.txt')
    file_LY = os.path.join(args.result_root, n + '_LY.txt')

    data_H = np.loadtxt(file_H, dtype=int, delimiter=',')
    data_W = np.loadtxt(file_W, dtype=int, delimiter=',')
    data_LX = np.loadtxt(file_LX, dtype=int, delimiter=',')
    data_LY = np.loadtxt(file_LY, dtype=int, delimiter=',')

    max_frame = len(data_H)

    for i in range(1, max_frame+1):
        image_path = os.path.join(image_folder, 'img{0:05}.jpg'.format(i))

        image = cv2.imread(image_path)

        # get all the rectangles
        all_H = data_H[i-1, :]
        all_W = data_W[i - 1, :]
        all_LX = data_LX[i - 1, :]
        all_LY = data_LY[i - 1, :]

        ids = []
        for j in range(len(all_H)):
            if all_H[j] > 0:
                ids += [j]

        # draw rectangles
        for id in ids:
            h = all_H[id]
            w = all_W[id]
            x = all_LX[id]
            y = all_LY[id]
            if id not in colors:
                colors[id] = tuple((np.random.rand(3) * 255).astype(int).tolist())

            image = cv2.rectangle(image, (x, y), (x+w, y+h), colors[id], 2)

        cv2.imshow("ua show", image)
        cv2.waitKey(25)
