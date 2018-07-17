from layer.sst import build_sst
from config.config import config
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

class TrackerConfig:

    max_record_frame = 25
    max_track_age = 25
    max_track_node = 25
    max_draw_track_node = 25

    sst_model_path = config['resume']
    cuda = config['cuda']
    mean_pixel = config['mean_pixel']
    image_size = (config['sst_dim'], config['sst_dim'])

    min_iou_frame_gap = [1, 2, 3]
    min_iou = [pow(0.2, i) for i in min_iou_frame_gap]

    min_merge_threshold = 0.1

    max_bad_node = 0.9

    decay = 0.9995

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
        self.all_iou = {}

    def update(self, sst, frame_index, features, boxes):
        # if the coming frame in the new frame
        if frame_index not in self.all_frame_index:
            # if the recorder have reached the max_record_frame.
            if len(self.all_frame_index) == self.max_record_frame:
                del_frame = self.all_frame_index[0]
                del self.all_features[del_frame]
                del self.all_boxes[del_frame]
                del self.all_similarity[del_frame]
                del self.all_iou[del_frame]
                self.all_frame_index = self.all_frame_index[1:]

            # add new item for all_frame_index, all_features and all_boxes. Besides, also add new similarity
            self.all_frame_index = np.append(self.all_frame_index, frame_index)
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes

            self.all_similarity[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                delta = pow(TrackerConfig.decay, frame_index - pre_index)
                pre_similarity = sst.forward_stacker_features(Variable(self.all_features[pre_index]), Variable(features), fill_up_column=False)
                self.all_similarity[frame_index][pre_index] = pre_similarity*delta

            self.all_iou[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                self.all_iou[frame_index][pre_index] = iou
        else:
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes
            index = self.all_frame_index.__index__(frame_index)

            for pre_index in self.all_frame_index[:index+1]:
                if pre_index == self.all_frame_index[-1]:
                    continue

                pre_similarity = sst.forward_stacker_features(Variable(self.all_features[pre_index]), Variable(self.all_features[-1]))
                self.all_similarity[frame_index][pre_index] = pre_similarity

                iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                self.all_similarity[frame_index][pre_index] = iou

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
        for i in range(len(self.s)):
            self.s[i, i] = -1

        self.f = np.array([], dtype=int)  # recorded frame
        self.uv = np.zeros((TrackerConfig.max_track_node, TrackerConfig.max_track_node), dtype=int) # the box index
        self.id = Track._id_pool
        Track._id_pool += 1
        self.age = 0
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())
        self.valid = True

    def update(self, frame_index, similarity, index):
        if len(self.f) == TrackerConfig.max_track_node:
            # remove the first item
            self.f = self.f[1:]
            s = np.zeros((TrackerConfig.max_track_node, TrackerConfig.max_track_node), dtype=np.float)
            for i in range(len(self.s)):
                self.s[i, i] = -1
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

    def get_total_similarity(self):
        total_similarity = []
        total_count = 0
        for i, f in enumerate(self.f):
            if self.s[i, i] > 0:
                total_similarity += [self.s[i, i]]
                total_count += 1

        if total_count == 0:
            return 0
        else:
            return np.min(total_similarity)

    def remove_similarity_node(self, t):
        # Remove all the node with same id compared with t.
        remove_frame = []
        for i, f in enumerate(t.f):
            if f in self.f and t.uv[i, i] != -1:
                id1 = t.uv[i, i]
                index = np.where(self.f==f)[0][0]
                id = self.uv[index, index]
                if id == id1:
                    remove_frame += [index]
                    self.uv[index, index] = -1

        # Remove extra one node which is the cause of wrong matching.
        if len(remove_frame) > 0:
            min_index = min(remove_frame)
            min_index -= 1
            while min_index > 0:
                if self.uv[min_index, min_index] != -1:
                    self.uv[min_index, min_index] = -1
                min_index -= 1


        # if remove all the nodes, then remove this track
        is_valid = False
        for i, f in enumerate(self.f):
            if self.uv[i, i] != -1:
                is_valid = True
                break

        self.valid = is_valid

    def get_bad_probability(self):
        if len(self.f) == 0:
            return 0.0

        bad_num = 0
        for i, f in enumerate(self.f):
            if self.uv[i, i] == -1:
                bad_num += 1.0

        return bad_num / len(self.f)



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

    @staticmethod
    def get_iou(pre_boxes, next_boxes):
        h = len(pre_boxes)
        w = len(next_boxes)
        if h == 0 or w == 0:
            return []

        iou = np.zeros((h, w), dtype=float)
        for i in range(h):
            b1 = np.copy(pre_boxes[i, :])
            b1[2:] = b1[:2] + b1[2:]
            for j in range(w):
                b2 = np.copy(next_boxes[j, :])
                b2[2:] = b2[:2] + b2[2:]
                overlap = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3])-max(b1[1], b2[1]), 0)
                area = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - overlap
                iou[i,j] = overlap / area

        return iou

    @staticmethod
    def get_merge_value(t1, t2):
        merge_value = 0
        if t1 is t2:
            return merge_value

        for i, f1 in enumerate(t1.f):
            for j, f2 in enumerate(t2.f):
                if f1 == f2 and t1.uv[i, i] == t2.uv[j, j] and t1.uv[i, i] != -1:
                    merge_value += 1
        return merge_value / float(TrackerConfig.max_track_node)

    @staticmethod
    def merge(t1, t2):
        # keep the track with the highest matching probability.
        # remove the overlapped node of the bad one
        s1 = t1.get_total_similarity()
        s2 = t2.get_total_similarity()

        is_t1 = False
        if s1 == 0 and s2 == 0:
            if t1.id > t2.id:
                is_t1 = True
        else:
            if s1 < s2:
                is_t1 = True

        if is_t1:
            t1.remove_similarity_node(t2)
        else:
            t2.remove_similarity_node(t1)


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

    def get_similarity_uv_by_linear_assignment(self, t, frame_index):
        res_similarity = []
        res_uv = []

        # for i, f in enumerate(t.f):
        #     if len(t.f) ==

    def get_similarity_uv(self, t, frame_index):
        res_similarity = []
        res_uv = []
        for i, f in enumerate(t.f):
            if len(t.f) == TrackerConfig.max_track_node and i == 0:
                continue

            all_iou = self.recorder.all_iou[frame_index][f]
            all_similarity = self.recorder.all_similarity[frame_index][f]
            selected_box_index = t.uv[i, i]
            if selected_box_index == -1: # cannot find box in f frame.
                res_similarity += [0]
                res_uv += [-1]
                continue

            # combine the similarity with the iou
            selected_similarity = np.copy(all_similarity[selected_box_index, :])
            delta_f = frame_index - f
            if delta_f in TrackerConfig.min_iou_frame_gap:
                iou_index = TrackerConfig.min_iou_frame_gap.index(delta_f)
                selected_iou = (all_iou[selected_box_index, :] >= TrackerConfig.min_iou[iou_index]).astype(float)
                selected_iou = np.append(selected_iou, 1.0)
                selected_similarity = selected_similarity * selected_iou

            max_index = np.argmax(selected_similarity)
            max_value = all_similarity[selected_box_index, max_index]

            if max_index == all_similarity.shape[1] - 1: # new node
                max_index = -1
            res_uv += [int(max_index)]
            res_similarity += [float(max_value)]

        # get the representation box of this frame.
        res = {}
        for uv, s in zip(res_uv, res_similarity):
            # if s < 0.5:
            #     continue
            if uv not in res:
                res[uv] = [s]
            else:
                res[uv] += [s]

        if len(res.keys()) > 0:
            max_uv = max(res.keys(), key=(lambda k: np.sum(res[k])))
        else:
            max_uv = -1

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
                start = len(nodes) - self.max_drawing_track
            else:
                start = 0

            for n1, n2 in zip(nodes[start:], nodes[start + 1:]):
                c1 = (int((n1[0] + n1[2] / 2.0) * w), int((n1[1] + n1[3]) * h))
                c2 = (int((n2[0] + n2[2] / 2.0) * w), int((n2[1] + n2[3]) * h))
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
                record_id += uv[-1:]  # record the representation
                t.update(frame_index, similarity, uv)

            # add new tracks
            for i in range(len(detection)):
                if i not in record_id:
                    t = Track()
                    t.update(frame_index, [1], [i])
                    self.tracks.append(t)

        # merge tracks
        l_track = len(self.tracks)
        if l_track != 0:
            merge_matrix = np.zeros((l_track, l_track), dtype=float)
            for i, t1 in enumerate(self.tracks):
                for j, t2 in enumerate(self.tracks):
                    merge_matrix[i, j] = TrackUtil.get_merge_value(t1, t2)

            merge_matrix = merge_matrix > TrackerConfig.min_merge_threshold
            if sum(sum(merge_matrix))  > 0:
                for i in range(l_track):
                    if self.tracks[i].valid:
                        for j in range(l_track):
                            if self.tracks[j].valid and merge_matrix[i, j]:
                                TrackUtil.merge(self.tracks[i], self.tracks[j])

            self.tracks = [t for t in self.tracks if t.valid]

        # remove older track
        self.tracks = [t for t in self.tracks if t.age < TrackerConfig.max_track_age and t.get_bad_probability() < TrackerConfig.max_bad_node]

        self.frame_index += 1

        if show_image:
            image = self.show(image)
            return image
