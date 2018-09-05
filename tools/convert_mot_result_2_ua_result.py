import argparse
import os
import numpy as np

print('''
Usage: convert_mat_2_ua --ua="ua root path"
''')


parser = argparse.ArgumentParser(description='UA Result Formatter')
parser.add_argument('--mot_folder', default=r"D:\ssj\DETRAC\20170721Result\0812-E25-M80-G30-TestSet-EB-MOT",
                    help='''mot result folder, with the following directory structure:
                    folder
                    |
                    |-- 0.1
                    |-- 0.2
                    |-- ...
                    ''')
parser.add_argument('--ua_folder', default=r"D:\ssj\DETRAC\20170721Result\0812-E25-M80-G30-TestSet-EB-UA", help='ua result folder. This tool would create this folder with same sturcture')

args = parser.parse_args()


class ConvertTools:
    @staticmethod
    def init(mot_folder, ua_folder):
        if not os.path.exists(mot_folder):
            raise FileNotFoundError('cannot find {}'.format(mot_folder))

        if not os.path.exists(ua_folder):
            os.mkdir(ua_folder)


        # get all the videos' frame number from 0.0 folder
        frame_number = {}
        path00 = os.path.join(mot_folder, '0.0')
        files00 = os.listdir(path00)
        for f in files00:
            if 'speed' not in f:
                frame_number[os.path.splitext(f)[0]] = max(np.loadtxt(os.path.join(path00, f), dtype=int)[:, 0])



        # get all directory in mot_folder path
        threshold_folder = [os.path.join(mot_folder, d) for d in os.listdir(mot_folder)]
        threshold_folder = list(filter(lambda f: os.path.isdir(f), threshold_folder))
        for folder in threshold_folder:
            ua_threshold_folder = os.path.join(ua_folder, os.path.basename(folder))
            if not os.path.exists(ua_threshold_folder):
                os.mkdir(ua_threshold_folder)
            # list all the files
            files = [os.path.join(folder, f) for f in os.listdir(folder)]
            files = sorted(list(filter(lambda f: os.path.isfile(f) and 'speed' not in os.path.basename(f), files)))
            for file in files:
                print('process: {}====>'.format(file))
                ua_file = os.path.join(ua_threshold_folder, os.path.splitext(os.path.basename(file))[0]) + "_{}.txt"
                data = np.loadtxt(file, dtype=int)
                if len(data) == 0:
                    np.savetxt(ua_file.format('LX'), [], fmt='%i')
                    np.savetxt(ua_file.format('LY'), [], fmt='%i', )
                    np.savetxt(ua_file.format('W'), [], fmt='%i')
                    np.savetxt(ua_file.format('H'), [], fmt='%i')
                    np.savetxt(ua_file.format('speed'), [], fmt='%f')
                    continue
                if len(data.shape) == 1:
                    data = np.expand_dims(data, axis=0)

                data[:, 0] = data[:, 0] - 1
                data[:, 1] = data[:, 1] - 1
                # max_f = max(data[:, 0])+1
                max_f = frame_number[os.path.splitext(os.path.basename(file))[0]]
                time = np.loadtxt(os.path.splitext(file)[0]+'-speed.txt', dtype=float)
                if time == 0:
                    speed = 0
                else:
                    speed = max_f / time
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
                # if len(data) > 1:
                #     ua_data_LX[0, :] = ua_data_LX[1, :]
                #     ua_data_LY[0, :] = ua_data_LY[1, :]
                #     ua_data_W[0, :] = ua_data_W[1, :]
                #     ua_data_H[0, :] = ua_data_H[1, :]

                np.savetxt(ua_file.format('LX'), ua_data_LX, fmt='%i')
                np.savetxt(ua_file.format('LY'), ua_data_LY, fmt='%i', )
                np.savetxt(ua_file.format('W'), ua_data_W, fmt='%i')
                np.savetxt(ua_file.format('H'), ua_data_H, fmt='%i')
                np.savetxt(ua_file.format('speed'), [speed], fmt='%f')

        # save sequence name file
        sequenceNames = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(threshold_folder[0])]
        np.savetxt(os.path.join(ua_folder, 'sequences.txt'), sequenceNames, fmt='%s', delimiter='\n')

        # save threshold file name
        threshold = os.listdir(mot_folder)
        np.savetxt(os.path.join(ua_folder, 'thresh.txt'), threshold, fmt='%s',delimiter='\n')

if __name__ == '__main__':
    # condition
    ConvertTools.init(args.mot_folder, args.ua_folder)
