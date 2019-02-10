'''
author: shijie Sun
email: shijieSun@chd.edu.cn
'''
import argparse
import os
import scipy.io as scio
import numpy as np

print('''
Usage: convert_mat_2_ua --ua="ua root path"
''')

parser = argparse.ArgumentParser(description='Convert ')
parser.add_argument('--root', default="/media/jianliu/ssm/dataset/dataset/UA-DETRAC", help='UA-DETRAC data set root directory, such as ua, we will create one directory called gt')
parser.add_argument('--annotations_mat', default="DETRAC-Train-Annotations-MAT", help='mat folder name')

args = parser.parse_args()

class ConvertMat2UA:
    @staticmethod
    def run(root, mat_folder_name):
        print('read files >>>>>>>>>>>>>')
        if not os.path.exists(root):
            raise FileNotFoundError('cannot find file ' + root)

        mat_folder = os.path.join(root, mat_folder_name)

        if not os.path.exists(mat_folder):
            raise FileNotFoundError('cannot find file ' + mat_folder)

        print('create gt folder')

        gt_folder = os.path.join(root, 'gt')
        if not os.path.exists(gt_folder):
            os.mkdir(gt_folder)

        print('search mat')

        mat_files_name = [f for f in os.listdir(mat_folder) if os.path.splitext(f)[1] == '.mat']
        mat_files_full_name = [os.path.join(mat_folder, f) for f in mat_files_name]

        for i, f in enumerate(mat_files_full_name):
            print('process :', f, '>>>>>')

            file_name = os.path.join(gt_folder, os.path.splitext(os.path.basename(f))[0]+'.txt')

            mat = scio.loadmat(f)['gtInfo'][0][0]

            X = mat[0]
            Y = mat[1]
            H = mat[2]
            W = mat[3]

            res = []
            for trackId, (xc, yc, hc, wc) in enumerate(zip(X.T, Y.T, H.T, W.T)):
                for frameId, (x, y, h, w) in enumerate(zip(xc, yc, hc, wc)):
                    if x != 0 and y != 0 and h!=0 and w!=0:
                        res += [[frameId, trackId, x-w/2.0, y-h, x+w/2.0, y]]

            res = np.array(res)

            np.savetxt(file_name, res, delimiter=',', fmt="%d,%d,%1.2f,%1.2f,%1.2f,%1.2f")

        print('=================Well Done=================')

if __name__ == '__main__':
    ConvertMat2UA.run(args.root, args.annotations_mat)



