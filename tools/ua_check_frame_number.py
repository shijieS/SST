'''
author: shijieSun
email: shijieSun@chd.edu.cn
'''
import argparse
import os
import numpy as np
from tools.utils.eb_utils import get_sequence_name_by_det_folder

print('''
Usage: ua_check_frame_number --detection_root="your detection folder" --result_root="your template folder such as GOG/DPM"
''')

parser = argparse.ArgumentParser(description='the tools for check ua result format')
parser.add_argument('--detection_root', default=r"D:\ssj\DETRAC\20170721Result\0812-E25-M80-G30-TestSet-EB-Upload\EB\EB", help='your result folder to check')
parser.add_argument('--result_root', default=r"D:\ssj\DETRAC\20170721Result\0812-E25-M80-G30-TestSet-EB-Upload\SST", help='the template root to check')

args = parser.parse_args()

sequences = get_sequence_name_by_det_folder(args.detection_root)

for seq in sequences:
    det = np.loadtxt(os.path.join(args.detection_root, '{}_Det_EB.txt'.format(seq)), dtype=float, delimiter=',')
    det_frame_num = np.max(det[:, 0])

    result_file = os.path.join(os.path.join(args.result_root, '0.0'), '{}_H.txt'.format(seq))
    if not os.path.exists(result_file):
        continue
    res = np.loadtxt(result_file, dtype=float)
    if len(res) != det_frame_num:
        print(seq)


