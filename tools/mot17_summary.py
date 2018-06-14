import pandas as pd
import numpy as np
from config.config import config
from utils.timer import Timer
import argparse
import os

parser = argparse.ArgumentParser(description='MOT 17 summary')
parser.add_argument('--mot_root', default=r'/home/ssm/ssj/dataset/MOT17', help='MOT ROOT')
args = parser.parse_args()

class MOT17Summary:
    def __init__(self, mot_root):
        self.mot_root = mot_root
        self.train_indexes = [2, 4, 5, 9, 10, 11, 13]
        self.test_indexes = [1, 3, 6, 7, 8, 12, 14]
        self.detectors = ['-DPM', '-FRCNN', '-SDP']

        # get all train det file name
        train_full_name_format = os.path.join(self.mot_root, 'train' + '/MOT17' + '-{:02}{}/det/det.txt')
        test_full_name_format = os.path.join(self.mot_root, 'test' + '/MOT17' + '-{:02}{}/det/det.txt')

        for i in self.train_indexes:
            for d in self.detectors:
                det_file = train_full_name_format.format(i, d)
                det = pd.read_csv(det_file, header=None)
                det = det[det[6] > 0.3]
                print('{}\t{}\t{}'.format(i, d, len(det)))

if __name__ == '__main__':
    s = MOT17Summary(args.mot_root)
