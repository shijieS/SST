import argparse
import os
import shutil
import scipy.io as scio
import numpy as np

print('''
Usage: extract_ua_result
''')

parser = argparse.ArgumentParser(description='the tools for extract txt file from result')
parser.add_argument('--result_root', default="/media/ssm/seagate/weights/UA-DETRAC/0812-E25-M80-G30-TestSet-EB", help='folder with result files')
parser.add_argument('--configure_str', default="5_0_4_1_5_5", help="The configure string")
parser.add_argument('--save_folder', default="/media/ssm/seagate/weights/UA-DETRAC/0812-E25-M80-G30-TestSet-EB-MOT", help='folder which needed to be extracted to')


args = parser.parse_args()


# create the save folder
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# check the result_root exists
if not os.path.exists(args.result_root):
    raise FileNotFoundError()

# get the subfolder of saved folder
subfolder_name = os.listdir(args.result_root)
subfolder_path = [os.path.join(os.path.join(args.result_root, f), args.configure_str) for f in subfolder_name]

savefolder_path = [os.path.join(args.save_folder, f) for f in subfolder_name]

# for each subfolder, copy all the text file in it.
for result_folder, save_folder in  zip(subfolder_path, savefolder_path):
    txt_names = [f for f in os.listdir(result_folder) if os.path.splitext(f)[1]=='.txt']
    txt_paths = [os.path.join(result_folder, f) for f in txt_names]
    save_paths = [os.path.join(save_folder, f) for f in txt_names]
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for file_from, file_to in zip(txt_paths, save_paths):
        shutil.copy(file_from, file_to)




