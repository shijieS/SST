'''
author: shijieSun
email: shijieSun@chd.edu.cn
'''
import argparse
import os
import numpy as np

print('''
Usage: ua_check_det_track --det_root="ua detection root" --track_root="your track result root"
''')

parser = argparse.ArgumentParser(description='check the consistency between detection and track')

parser.add_argument('--track_root', default=r"F:\dataset\dataset\UA-DETRAC\other-methods-result\Tracker-joint-det-track-id\Tracker-joint-det-track-id\Detector-joint-det-track-id", help='your track result root')

parser.add_argument('--save_root',default=r"F:\dataset\dataset\UA-DETRAC\other-methods-result\Tracker-joint-det-track-id\Detection" )


args = parser.parse_args()


print("""Start checking============>>>>>>>
""")

print("""get threshold of your track result""")
thresholds = [f for f in os.listdir(args.track_root) if os.path.isdir(os.path.join(args.track_root, f))]

print(thresholds)

if not os.path.exists(args.save_root):
    os.mkdir(args.save_root)

print("""Get sequences""")
sequences = np.unique([seq[:9] for seq in os.listdir(os.path.join(args.track_root, thresholds[0])) if os.path.splitext(seq)[1]==".txt"])

print(sequences)

print("""start analysis""")

for seq in sequences:
    print('start analysis sequence {}'.format(seq))
    save_path = os.path.join(args.save_root, seq + '_Det_EB.txt')

    record = {}
    new_record = {}
    for thresh in thresholds:
        print(thresh)
        track_H_path = os.path.join(os.path.join(args.track_root, thresh), seq+"_H.txt")
        track_W_path = os.path.join(os.path.join(args.track_root, thresh), seq + "_W.txt")
        track_LX_path = os.path.join(os.path.join(args.track_root, thresh), seq + "_LX.txt")
        track_LY_path = os.path.join(os.path.join(args.track_root, thresh), seq + "_LY.txt")
        H = np.loadtxt(track_H_path, dtype=float, delimiter=',').astype(int)
        W = np.loadtxt(track_W_path, dtype=float, delimiter=',').astype(int)
        LX = np.loadtxt(track_LX_path, dtype=float, delimiter=',').astype(int)
        LY = np.loadtxt(track_LY_path, dtype=float, delimiter=',').astype(int)

        # generate ne w rectangles
        for frame_index, (rowH, rowW, rowLX, rowLY) in enumerate(zip(H, W, LX, LY)):
            ids = []
            if type(rowH) == np.int32:
                continue
            for i, h in enumerate(rowH):
                if h > 0:
                    ids += [i]
            for id in ids:
                record[(frame_index+1, rowLX[id], rowLY[id], rowW[id], rowH[id])] = float(thresh)+0.05

    data = []
    for thresh in thresholds:
        track_H_path = os.path.join(os.path.join(args.track_root, thresh), seq+"_H.txt")
        track_W_path = os.path.join(os.path.join(args.track_root, thresh), seq + "_W.txt")
        track_LX_path = os.path.join(os.path.join(args.track_root, thresh), seq + "_LX.txt")
        track_LY_path = os.path.join(os.path.join(args.track_root, thresh), seq + "_LY.txt")
        H = np.loadtxt(track_H_path, dtype=float, delimiter=',').astype(int)
        W = np.loadtxt(track_W_path, dtype=float, delimiter=',').astype(int)
        LX = np.loadtxt(track_LX_path, dtype=float, delimiter=',').astype(int)
        LY = np.loadtxt(track_LY_path, dtype=float, delimiter=',').astype(int)

        # generate new rectangles
        for frame_index, (rowH, rowW, rowLX, rowLY) in enumerate(zip(H, W, LX, LY)):
            if type(rowH) == np.int32:
                continue
            ids = []
            for i, h in enumerate(rowH):
                if h > 0:
                    ids += [i]
            for i, id in enumerate(ids):
                t = record[(frame_index+1, rowLX[id], rowLY[id], rowW[id], rowH[id])]
                if t < 0:
                    continue
                if frame_index+1 not in new_record:
                    new_record[frame_index+1] = {}
                new_record[frame_index+1][(rowLX[id], rowLY[id], rowW[id], rowH[id])] = t
                record[(frame_index + 1, rowLX[id], rowLY[id], rowW[id], rowH[id])] = -1


    for frame_index in sorted(new_record.keys()):
        for id, (x, y, w, h) in enumerate(new_record[frame_index].keys()):
            t = new_record[frame_index][(x, y, w, h)]
            data += [[frame_index, id+1, x, y, w, h, t]]

    np.savetxt(save_path, data, fmt='%d,%d,%1.2f,%1.2f,%1.2f,%1.2f,%1.2f', delimiter=',')









