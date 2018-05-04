from layer.sst import build_sst
from config.config import config, init_tracker_config
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

class TrackerConfig:

    max_record_frame = 10
    max_track_age = 10
    max_track_node = 5
    max_draw_track_node = 5

    sst_model_path = config['resume']
    cuda = config['cuda']
    mean_pixel = config['mean_pixel']
    image_size = (config['sst_dim'], config['sst_dim'])


class FeatureRecorder:
    '''
    Record features and boxes every frame
    '''

    def __init__(self):
        self.max_record_frame = TrackerConfig.max_record_frame
        self.all_frame_index = np.array([], dtype=int)
        self.all_features = {}
        self.all_boxes = {}
        self.all_similarity = {}

    def update(self, sst, frame_index, features, boxes):
        # if the coming frame in the new frame
        if frame_index not in self.all_frame_index:
            # if the recorder have reached the max_record_frame.
            if len(self.all_frame_index) == self.max_record_frame:
                del_frame = self.all_frame_index[0]
                del self.all_features[del_frame]
                del self.all_boxes[del_frame]
                del self.all_similarity[del_frame]
                self.all_frame_index = self.all_frame_index[1:]

            # add new item for all_frame_index, all_features and all_boxes. Besides, also add new similarity
            self.all_frame_index = np.append(self.all_frame_index, frame_index)
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes

            self.all_similarity[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                pre_similarity = sst.forward_stacker_features(Variable(self.all_features[pre_index]), Variable(features), fill_up_column=False)
                self.all_similarity[frame_index][pre_index] = pre_similarity
        else:
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes
            index = self.all_frame_index.__index__(frame_index)

            for pre_index in self.all_frame_index[:index+1]:
                if pre_index == self.all_frame_index[-1]:
                    continue

                pre_similarity = sst.forward_stacker_features(Variable(self.all_features[pre_index]), self.all_frame_index[-1])
                self.all_similarity[frame_index][pre_index] = pre_similarity

    def get_feature(self, frame_index, detection_index):
        '''
        get the feature by the specified frame index and detection index
        :param frame_index: start from 0
        :param detection_index: start from 0
        :return: the corresponding feature at frame index and detection index
        '''

        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
            if len(features) == 0:
                return None
            if detection_index < len(features):
                return features[detection_index]

        return None

    def get_box(self, frame_index, detection_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
            if len(boxes) == 0:
                return None

            if detection_index < len(boxes):
                return boxes[detection_index]
        return None

    def get_features(self, frame_index):
        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
        else:
            return None
        if len(features) == 0:
            return None
        return features

    def get_boxes(self, frame_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
        else:
            return None

        if len(boxes) == 0:
            return None
        return boxes

class Node:
    def __init__(self, frame_index, feature_index, detection_index):
        self.frame_index = frame_index
        self.feature_index = feature_index
        self.detection_index = detection_index
        self.active = True

class Track:
    _id_pool = 0

    def __init__(self):
        self.s = np.zeros((TrackerConfig.max_track_node, TrackerConfig.max_track_node), dtype=np.float) # similarity score
        self.f = np.array([], dtype=int)  # recorded frame
        self.uv = np.zeros((TrackerConfig.max_track_node, TrackerConfig.max_track_node), dtype=int) # the box index
        self.id = Track._id_pool
        Track._id_pool += 1
        self.age = 0
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())

    def update(self, frame_index, similarity, index):
        if len(self.f) == TrackerConfig.max_track_node:
            # remove the first item
            self.f = self.f[1:]
            s = np.zeros((TrackerConfig.max_track_node, TrackerConfig.max_track_node), dtype=np.float)
            s[:-1, :-1] = self.s[1:, 1:]
            self.s = s
            uv = np.zeros((TrackerConfig.max_track_node, TrackerConfig.max_track_node), dtype=int)
            uv[:-1, :-1] = self.uv[1:, 1:]
            self.uv = uv

        self.f = np.append(self.f, frame_index)
        i = len(self.f) - 1
        self.s[:len(similarity), i] = similarity
        self.uv[:len(index), i] = index

    def add_age(self):
        self.age += 1

    def reset_age(self):
        self.age = 0

    def get_all_nodes(self, recorder):
        all_nodes = []
        for i, f in enumerate(self.f):
            id = self.uv[i, i]
            if id == -1:
                continue
            all_nodes.append(recorder.all_boxes[f][id, :])

        return all_nodes

    def get_current_box(self, recorder):
        if len(self.f) > 0 and self.age == 0:
            frame_index = self.f[-1]
            i = len(self.f) - 1
            id = self.uv[i, i]
            if id != -1:
                return recorder.all_boxes[frame_index][id, :]
        return None

class TrackUtil:
    @staticmethod
    def convert_detection(detection):
        '''
        transform the current detection center to [-1, 1]
        :param detection: detection
        :return: translated detection
        '''
        # get the center, and format it in (-1, 1)
        center = (2 * detection[:, 0:2] + detection[:, 2:4]) - 1.0
        center = torch.from_numpy(center.astype(float)).float()
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)

        if TrackerConfig.cuda:
            return Variable(center.cuda())
        return Variable(center)

    @staticmethod
    def convert_image(image):
        '''
        transform image to the FloatTensor (1, 3,size, size)
        :param image: same as update parameter
        :return: the transformed image FloatTensor (i.e. 1x3x900x900)
        '''
        image = cv2.resize(image, TrackerConfig.image_size).astype(np.float32)
        image -= TrackerConfig.mean_pixel
        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        image.unsqueeze_(dim=0)
        if TrackerConfig.cuda:
            return Variable(image.cuda())
        return Variable(image)


class TrackSet:
    def __init__(self):
        self.tracks = list()
        self.max_drawing_track = TrackerConfig.max_draw_track_node
        self.cuda = TrackerConfig.cuda
        self.recorder = FeatureRecorder()
        self.frame_index = 0

        # load the model
        self.sst = build_sst('test', 900)
        if self.cuda:
            cudnn.benchmark = True
            self.sst.load_state_dict(
                torch.load(TrackerConfig.sst_model_path)
            )
            self.sst = self.sst.cuda()
        else:
            self.sst.load_state_dict(torch.load(config['resume'], map_location='cpu'))
        self.sst.eval()

    def __getitem__(self, item):
        return self.tracks[item]

    def __len__(self):
        return len(self.tracks)

    def get_similarity_uv(self, t, frame_index):
        res_similarity = []
        res_uv = []
        for i, f in enumerate(t.f):
            if len(t.f) == TrackerConfig.max_track_node and i == 0:
                continue
            all_similarity = self.recorder.all_similarity[frame_index][f]
            selected_box_index = t.uv[i, i]
            if selected_box_index == -1: # cannot find box in f frame.
                res_similarity += [0]
                res_uv += [-1]
                continue

            max_index = np.argmax(all_similarity[selected_box_index, :])
            max_value = all_similarity[selected_box_index, max_index]
            if max_index == all_similarity.shape[1] - 1: # new node
                max_index = -1
            res_uv += [int(max_index)]
            res_similarity += [float(max_value)]

        # get the representation box of this frame.
        res = {}
        for uv, s in zip(res_uv, res_similarity):
            if uv not in res:
                res[uv] = s
            else:
                res[uv] += s
        print(frame_index)
        print(res)
        max_uv = max(res.keys(), key=(lambda k: res[k]))
        res_similarity += [1]
        res_uv += [max_uv]

        if max_uv == -1:
            t.age += 1
        else:
            t.age = 0

        return res_similarity, res_uv


    def show(self, image):
        h, w, _ = image.shape

        # draw rectangle
        for t in self.tracks:
            b = t.get_current_box(self.recorder)
            if not b is None:
                image = cv2.putText(image, str(t.id), (int(b[0] * w), int((b[1]) * h)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    t.color, 3)
                image = cv2.rectangle(image, (int(b[0] * w), int((b[1]) * h)),
                                      (int((b[0] + b[2]) * w), int((b[1] + b[3]) * h)), t.color, 2)

        # draw line
        for t in self.tracks:
            if t.age > 1:
                continue
            nodes = t.get_all_nodes(self.recorder)
            if len(nodes) > self.max_drawing_track:
                start = len(t.nodes) - self.max_drawing_track
            else:
                start = 0
            for n1, n2 in zip(nodes[start:], nodes[start + 1:]):
                c1 = (int((n1[0] + n1[2] / 2.0) * w), int((n1[1] + n1[3]) * h))
                c2 = (int((n2[0] + n1[2] / 2.0) * w), int((n2[1] + n2[3]) * h))
                image = cv2.line(image, c1, c2, t.color, 2)

        return image



    def update(self, image, detection, show_image):
        '''
        1. get all the detection features and update the feature recorder
        2. according the features update trackset
        :param image: the current frame
        :param detection: the detected boxes
        :return: update current track according to the extracted features
        '''
        frame_index = self.frame_index
        input_image = TrackUtil.convert_image(image)
        input_detection = TrackUtil.convert_detection(detection)

        features = self.sst.forward_feature_extracter(input_image, input_detection)
        self.recorder.update(self.sst, frame_index, features.data, detection)

        if frame_index == 0:
            for i in range(len(detection)):
                t = Track()
                t.update(0, [1], [i])
                self.tracks.append(t)

        else:
            # get all similarity between each frame
            record_id = []
            for t in self.tracks:
                # get every boxes in current frame's similarity.
                similarity, uv = self.get_similarity_uv(t, frame_index)
                record_id += uv[-1:]  # record the respresentation
                t.update(frame_index, similarity, uv)

            # add new tracks
            for i in range(len(detection)):
                if i not in record_id:
                    t = Track()
                    t.update(frame_index, [1], [i])
                    self.tracks.append(t)

        # remove older track
        self.tracks = [t for t in self.tracks if t.age < TrackerConfig.max_track_age]

        self.frame_index += 1

        if show_image:
            image = self.show(image)
            return image
