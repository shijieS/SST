import argparse
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from layer.sst import build_sst
import numpy as np

print('''
Usage: compare_tools --image1="" image2="" model_path=""

You can draw any rectangle on the gui.
press the follows button for more function:
'd': delete the latest rectangle
'D': delete all the rectangles
'c': calculate the similarity matrix
''')

parser = argparse.ArgumentParser(description='Single Shot Joint Tracker Train')
parser.add_argument('--image1', default="/media/jianliu/ssm/dataset/dataset/MOT/17/MOT17/train/MOT17-11-FRCNN/img1/000001.jpg", help='Previous Image')
parser.add_argument('--image2', default="/media/jianliu/ssm/dataset/dataset/MOT/17/MOT17/train/MOT17-11-FRCNN/img1/000030.jpg", help='Current Image')
parser.add_argument('--model_path', default="/media/jianliu/ssm/ssj/github/weights/sst300_0712_66400.pth", help='sst net model path')
parser.add_argument('--cuda', default=True, help="use gpu or not")

args = parser.parse_args()

class CompareTools:
    img1 = None
    img2 = None
    img1_convert = None
    img2_convert = None
    img = None
    org_img = None
    sst = None
    cuda = False
    resize_rate = 0.4
    drawing = False

    save_objects = {'rect' : [], 'text': []}

    @staticmethod
    def init(img1_path, img2_path, model_path, cuda):
        print('start init >>>>>>>>>>>>>>')
        if not os.path.exists(img1_path) or not os.path.exists(img2_path) or not os.path.exists(model_path):
            raise ValueError("input parameter not right")

        CompareTools.cuda = cuda

        print('load image...')
        # load image
        CompareTools.img1 = cv2.imread(img1_path)
        CompareTools.img2 = cv2.imread(img2_path)
        CompareTools.img1_convert = CompareTools.convert_image(CompareTools.img1, CompareTools.cuda)
        CompareTools.img2_convert = CompareTools.convert_image(CompareTools.img2, CompareTools.cuda)
        CompareTools.img = np.concatenate([CompareTools.img1, CompareTools.img2], axis=0)
        CompareTools.img_org = np.copy(CompareTools.img)

        print('load model...')
        # load net
        CompareTools.sst = build_sst('test', 900, CompareTools.cuda)
        if cuda:
            cudnn.benchmark = True
            CompareTools.sst.load_state_dict(
                torch.load(model_path)
            )
            CompareTools.sst = CompareTools.sst.cuda()
        else:
            CompareTools.sst.load_state_dict(torch.load(model_path))

        print('finish init <<<<<<<<<<<<')
    @staticmethod
    def convert_image(image, cuda):
        '''
        transform image to the FloatTensor (1, 3,size, size)
        :param image: same as update parameter
        :return: the transformed image FloatTensor (i.e. 1x3x900x900)
        '''
        image = cv2.resize(image, (900, 900)).astype(np.float32)
        image -= (104, 117, 123)
        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        image.unsqueeze_(dim=0)
        if cuda:
            return Variable(image.cuda())
        return Variable(image)

    @staticmethod
    def convert_boxes(boxes):
        boxes = np.array(boxes)
        center = (boxes[:, 0:2] + boxes[:, 2:4]) - 1.0
        center = torch.from_numpy(center.astype(float)).float()
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)

        if CompareTools.cuda:
            return Variable(center.cuda())
        return Variable(center)

    @staticmethod
    def select_object(event, x, y, flag, param):
        global ix, iy

        color = tuple((np.random.rand(3) * 255).astype(int).tolist())

        if event == cv2.EVENT_LBUTTONDOWN:
            CompareTools.drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if CompareTools.drawing == True:
                cv2.rectangle(CompareTools.img, (ix, iy), (x, y), color, 2)

        elif event == cv2.EVENT_LBUTTONUP:
            CompareTools.drawing = False
            cv2.rectangle(CompareTools.img, (ix, iy), (x, y), color, 2)

            CompareTools.save_objects['rect'] += [(
                (ix, iy),
                (x, y),
                str(len(CompareTools.save_objects['rect'])),
                tuple((np.random.rand(3) * 255).astype(int).tolist())
            )]

    @staticmethod
    def draw(img):
        boxes = CompareTools.save_objects['rect']
        for b in boxes:
            start = (int(b[0][0] / CompareTools.resize_rate), int(b[0][1] / CompareTools.resize_rate))
            end = (int(b[1][0] / CompareTools.resize_rate), int(b[1][1] / CompareTools.resize_rate))
            text = b[2]
            cv2.rectangle(img, start, end, b[3], 3)
            cv2.putText(img, text, start, cv2.FONT_HERSHEY_SIMPLEX, 1, b[3], 2)
        return img

    @staticmethod
    def get_similarity():
        h, w, _ = CompareTools.img1.shape
        boxes = [list(b[0])+list(b[1]) for b in CompareTools.save_objects['rect']]

        def convert_box(x):
            x[0] /= float(w) * CompareTools.resize_rate
            x[2] /= float(w) * CompareTools.resize_rate
            x[1] /= float(h) * CompareTools.resize_rate
            x[3] /= float(h) * CompareTools.resize_rate
            return x

        boxes = list(map(convert_box, boxes))
        boxes1 = [b for b in boxes if b[1] < 1]
        boxes2 = [[b[0], b[1]-1, b[2], b[3]-1] for b in boxes if b[1] >= 1]

        boxes1 = CompareTools.convert_boxes(boxes1)
        boxes2 = CompareTools.convert_boxes(boxes2)

        return CompareTools.sst.get_similarity(CompareTools.img1_convert, boxes1, CompareTools.img2_convert, boxes2)

    @staticmethod
    def run():
        print('''
        select the rectangle, and press the following key for more function:
        'd': delete the latest rectangle
        'D': delete all the rectangles
        'c': calculate the similarity matrix
        ''')
        h, w, _ = CompareTools.img.shape
        # start draw
        title = "images(left is previous, right is current)"
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, CompareTools.select_object)

        while(1):
            CompareTools.img = CompareTools.draw(np.copy(CompareTools.img_org))
            CompareTools.img = cv2.resize(CompareTools.img, (int(CompareTools.resize_rate*w), int(CompareTools.resize_rate*h)))
            cv2.imshow(title, CompareTools.img)
            key = cv2.waitKey(20)
            if key == 100:      #    'd'
                CompareTools.save_objects['rect'] = CompareTools.save_objects['rect'][:-1]
                print('delete the latest box!')
            elif key == 68:     #   'D'
                CompareTools.save_objects['rect'] = []
                print('delete all boxes')
            elif key == 99:    #   'c'
                print('start calculate the similarity!')
                s = CompareTools.get_similarity()
                print(s)
            if key & 0xFF == 27:
                break


if __name__ == '__main__':
    # condition
    CompareTools.init(args.image1, args.image2, args.model_path, args.cuda)
    CompareTools.run()



