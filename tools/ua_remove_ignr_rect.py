'''
author: shijie Sun
email: shijieSun@chd.edu.cn
'''

import cv2
import argparse
import os
import numpy as np

print('''
Remove rectangles in ignore region
Usage: ua_remove_ignr_rect --detection_root="your detection folder" --ignore_root="your ignore folder" --save_root="the saved folder"
''')

parser = argparse.ArgumentParser(description='the tool for removing rectangles in ignore region')
parser.add_argument('--detection_root', default=r"D:\ssj\DETRAC\20170721Result\0803-E25-M80-G30-TestSet-EB-UA\EB", help='the detection folder')
parser.add_argument('--ignore_root', default=r"F:\ssj\github\DETRAC-MOT-toolkit\evaluation\igrs", help='the ignore folder')
parser.add_argument('--save_root', default=r"D:\ssj\DETRAC\20170721Result\0803-E25-M80-G30-TestSet-EB-UA\EB-new", help='the destination folder')


args = parser.parse_args()


def isInIgnore(rect, ignoreRect):
    '''
    judge whether rect in ignore region
    :param rect: rectangle
    :return: true if in ignore, or false if not
    '''
    for irec in ignore_rects:
        iL = irec[0]
        iT = irec[1]
        iR = irec[0] + irec[2]
        iB = irec[1] + irec[3]
        if  rect[0] > iL and rect[0] < iR and \
            rect[1] > iT and rect[1] < iB and \
            rect[0]+rect[2] > iL and rect[0]+rect[2] < iR and \
            rect[1]+rect[3] > iT and rect[1]+rect[3] < iB:
            return True

    return False

print('''
start processing ========>>>>>>>>>>
''')

print('''
Reading the results
''')


names = np.unique([os.path.splitext(os.path.basename(f))[0][:9] for f in os.listdir(args.detection_root)])

print('all the sequences: {}'.format(names))


for n in names:
    print('processing {}>>>>>>'.format(n))
    ignore_path = os.path.join(args.ignore_root, n+'_IgR.txt')
    detect_path = os.path.join(args.detection_root, n+'_Det_EB.txt')
    saved_path = os.path.join(args.save_root, n+'_Det_EB.txt')

    print('read content')
    detections = np.loadtxt(detect_path, dtype=float, delimiter=',')
    ignore_rects = np.loadtxt(ignore_path, dtype=float, delimiter=',')


    result = np.array(list(filter(lambda x: not isInIgnore(x[2:6], ignore_rects), detections)))

    np.savetxt(saved_path, result, fmt='%d,%d,%1.2f,%1.2f,%1.2f,%1.2f,%1.2f', delimiter=',')

