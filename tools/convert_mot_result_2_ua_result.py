import argparse
import os
import numpy as np

print('''
Usage: convert_mat_2_ua --ua="ua root path"
''')


parser = argparse.ArgumentParser(description='UA Result Formatter')
parser.add_argument('--mot_folder', default=r"F:\ssj\github\SST\result\DETRAC\mot",
                    help='''mot result folder, with the following directory structure:
                    folder
                    |
                    |-- 0.1
                    |-- 0.2
                    |-- ...
                    ''')
parser.add_argument('--ua_folder', default=r"F:\ssj\github\SST\result\DETRAC\detrac", help='ua result folder. This tool would create this folder with same sturcture')

args = parser.parse_args()


class ConvertTools:
    @staticmethod
    def init(mot_folder, ua_folder):
        if not os.path.exists(mot_folder):
            raise FileNotFoundError('cannot find {}'.format(mot_folder))

        if not os.path.exists(ua_folder):
            os.mkdir(ua_folder)

        # get all directory in mot_folder path
        threshold_folder = [os.path.join(mot_folder, d) for d in os.listdir(mot_folder)]
        threshold_folder = list(filter(lambda f: os.path.isdir(f), threshold_folder))
        for folder in threshold_folder:
            ua_threshold_folder = os.path.join(ua_folder, os.path.basename(folder))
            if not os.path.exists(ua_threshold_folder):
                os.mkdir(ua_threshold_folder)
            # list all the files
            files = [os.path.join(folder, f) for f in os.listdir(folder)]
            files = list(filter(lambda f: os.path.isfile(f), files))
            for file in files:
                print('process: {}====>'.format(file))
                data = np.loadtxt(file, dtype=int)
                data[:, 0] = data[:, 0] - 2
                data[:, 1] = data[:, 1] - 1
                max_f = max(data[:, 0])+1
                max_id = max(data[:, 1])+1
                ua_data_LX = np.zeros((max_f, max_id), dtype=int)
                ua_data_LY = np.zeros((max_f, max_id), dtype=int)
                ua_data_H = np.zeros((max_f, max_id), dtype=int)
                ua_data_W = np.zeros((max_f, max_id), dtype=int)

                for row in data:
                    r = row[0]
                    c = row[1]
                    ua_data_LX[r, c] = row[2]
                    ua_data_LY[r, c] = row[3]
                    ua_data_W[r, c] = row[4]
                    ua_data_H[r, c] = row[5]

                ua_file = os.path.join(ua_threshold_folder, os.path.splitext(os.path.basename(file))[0])+"_{}.txt"
                np.savetxt(ua_file.format('LX'), ua_data_LX, fmt='%i')
                np.savetxt(ua_file.format('LY'), ua_data_LY, fmt='%i', )
                np.savetxt(ua_file.format('W'), ua_data_W, fmt='%i')
                np.savetxt(ua_file.format('H'), ua_data_H, fmt='%i')

        # save sequence name file
        sequenceNames = [os.path.basename(f) for f in os.listdir(threshold_folder)]
        np.savetxt(os.path.join(ua_folder, 'sequences.txt'), sequenceNames, delimiter='\n')

        # save threshold file name
        threshold = os.path.basename(os.listdir(threshold_folder))
        np.savetxt(os.path.join(ua_folder, 'thresh.txt'), threshold, delimiter='\n')

if __name__ == '__main__':
    # condition
    ConvertTools.init(args.mot_folder, args.ua_folder)
