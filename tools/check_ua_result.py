'''
author: shijieSun
email: shijieSun@chd.edu.cn
'''
import argparse
import os
import numpy as np

print('''
Usage: check_ua_result --ua_root="ua root path" --template_root="your template folder such as GOG/DPM"
''')

parser = argparse.ArgumentParser(description='the tools for check ua result format')
parser.add_argument('--ua_root', default=r"D:\ssj\DETRAC\20170721Result\detrac", help='your result folder to check')
parser.add_argument('--template_root', default=r"D:\ssj\DETRAC\GOG\DPM", help='the template root to check')

args = parser.parse_args()
print('start check the threshold folder=======>')

threshold = [str(round(d*0.1, 2)) for d in range(11)]
threshold_path = [os.path.join(args.ua_root, t) for t in threshold]

for t in threshold_path:
    if not os.path.exists(t):
        print("error, cannot find {}".format(t))
        exit()

print('threshold is ok!')
print('start check the frame number=======>')
template_threshold_path = [os.path.join(args.template_root, t) for t in threshold]

error_list = []
for ua, temp in zip(threshold_path, template_threshold_path):
    ua_names = os.listdir(ua)
    for n in ua_names:
        if 'speed' in n:
            continue
        ua_file = os.path.join(ua, n)
        template_file = os.path.join(temp, n)
        len_ua_file = len(np.loadtxt(ua_file, dtype=int))
        len_template_file = len(np.loadtxt(template_file, dtype=int, delimiter=','))
        if len_template_file == 0 or len_ua_file == 0:
            continue
        if len_ua_file != len_template_file:
            print("error, frame number error for {}".format(ua_file))
            error_list += [ua_file]
        print("{} is ok".format(ua_file))

print('check the following files')
print(error_list)

print('succeed, finished!')



