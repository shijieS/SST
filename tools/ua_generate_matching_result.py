import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
import argparse

from data.ua import UATrainDataset
from config.config import config
from layer.sst import build_sst
from layer.sst_loss import SSTLoss
from utils.augmentations import SSJEvalAugment, collate_fn
import time
from utils.operation import show_circle, show_batch_circle_image
import cv2

# build the model
sst = build_sst('test', 900)
if config['cuda']:
    cudnn.benchmark = True
    sst.load_state_dict(torch.load(config['resume']))
    sst = sst.cuda()
else:
    sst.load_state_dict(torch.load(config['resume'], map_location='cpu'))
sst.eval()


dataset = UATrainDataset(config['ua_image_root'],
                         config['ua_detection_root'],
                         config['ua_ignore_root'],
                         SSJEvalAugment(
                             config['sst_dim'], config['mean_pixel']
                         ))

data_loader = data.DataLoader(dataset, config['batch_size'],
                              num_workers=config['num_workers'],
                              shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=False)

batch_iterator = iter(data_loader)

criterion = SSTLoss(config['cuda'])

if not os.path.exists(config['log_folder']):
    os.mkdir(config['log_folder'])

image_format = os.path.join(config['log_folder'], '{}.jpg')
for i in range(len(dataset)):
    img_pre, img_next, boxes_pre, boxes_next, labels, valid_pre, valid_next = \
        next(batch_iterator)

    if img_pre is None:
        continue

    if config['cuda']:
        img_pre = Variable(img_pre.cuda())
        img_next = Variable(img_next.cuda())
        boxes_pre = Variable(boxes_pre.cuda())
        boxes_next = Variable(boxes_next.cuda())
        valid_pre = Variable(valid_pre.cuda(), volatile=True)
        valid_next = Variable(valid_next.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

    else:
        img_pre = Variable(img_pre)
        img_next = Variable(img_next)
        boxes_pre = Variable(boxes_pre)
        boxes_next = Variable(boxes_next)
        valid_pre = Variable(valid_pre)
        valid_next = Variable(valid_next)
        labels = Variable(labels, volatile=True)

    out = sst(img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next)
    loss_pre, loss_next, loss_similarity, loss, accuracy_pre, accuracy_next, accuracy, predict_indexes = criterion(out, labels, valid_pre, valid_next)

    result_image = show_batch_circle_image(img_pre, img_next, boxes_pre, boxes_next, valid_pre, valid_next, predict_indexes, i)
    result_image = result_image[0, :].permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(image_format.format(i), result_image)
    print(i)
