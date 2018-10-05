import cv2
import argparse
import os

print('''
Usage: comm_split_frames_from_video --ua="ua root path"
''')
parser = argparse.ArgumentParser(description='Split Frames from video')
parser.add_argument('--video_path', default=r"D:\ssj\temp\video\MVI_40855.avi",
                    help="input video path")
parser.add_argument('--output_folder', default=r"D:\ssj\temp\video\MVI_40855", help="output folder")

args = parser.parse_args()
if not os.path.exists(args.video_path):
    raise FileNotFoundError("cannot find file:{}".format(args.video_path))

if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)

vc = cv2.VideoCapture(args.video_path)

image_file_format = os.path.join(args.output_folder, "{}.jpg")

frame_index = 0
retval = True
while retval:
    retval, frame = vc.read()
    cv2.imwrite(image_file_format.format(frame_index), frame)
    frame_index += 1
    print(frame_index)



