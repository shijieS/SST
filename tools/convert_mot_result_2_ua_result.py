import argparse
import os
import numpy as np

print('''
Usage: convert_mat_2_ua --ua="ua root path"
''')


parser = argparse.ArgumentParser(description='UA Result Formatter')
parser.add_argument('--mot_folder', default="/media/jianliu/ssm/dataset/dataset/UA-DETRAC",
                    help='''mot result folder, with the following directory structure:
                    folder
                    |
                    |-- 0.1
                    |-- 0.2
                    |-- ...
                    ''')
parser.add_argument('--ua_folder', default="", help='ua result folder. This tool would create this folder with same sturcture')


