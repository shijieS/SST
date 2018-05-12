from layer.sst import build_sst
from config.config import config, init_tracker_config
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


# vw = cv2.VideoWriter('MOT17-11-DPM-temp.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920, 1080))
class Node:
    '''
    The Node is the basic element of a track. it contains the following information:
    1) extracted feature (it'll get removed when it isn't active
    2) box (a box (l, t, r, b)
    3) label (active label indicating keeping the features)
    4) detection, the formated box
    '''
    inherit_rate = 0.0
    def __init__(self, box, feature, detection):
        self.box = box
        self.feature = feature  #shape (1, 1, 450)
        self.detection = detection #shape (1, 1, 1, 1, 2)
        self.active = True
    def deactiviate(self):
        # remove the feature when it get deactive
        self.active = False
        del self.feature
        del self.detection
        self.feature = None
        self.detection = None

    def inherit_features_(self, previous):
        self.feature = self.feature*(1-Node.inherit_rate) + previous.feature*Node.inherit_rate

    def __del__(self):
        del self.feature, self.detection

class Track:
    '''
    Track is the class of track. it contains all the node and manages the node. it contains the following information:
    1) all the nodes
    2) track id. it is unique it identify each track
    3) track pool id. it is a number to give a new id to a new track
    4) age. age indicates how old is the track
    5) max_age. indicates the dead age of this track
    '''
    _id_pool = 1
    ''' for mot
    _max_age = 40
    _max_num_node = 36
    _max_save_feature = 18
    '''
    '''for kitti
    '''
    _max_age = 10
    _max_num_node = 5
    _max_save_feature = 4
    def __init__(self):
        self.nodes = list()
        self.id = Track._id_pool
        Track._id_pool += 1
        self.age = 0
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())

    def __del__(self):
        for n in self.nodes:
            del n

    def add_age(self):
        self.age += 1

    def reset_age(self):
        self.age = 0

    def add_node(self, node):
        if len(self.nodes) > Track._max_save_feature + 1:
            self.nodes[-(Track._max_save_feature + 1)].deactiviate()
        self.nodes.append(node)
        self.reset_age()
        self._volatile_memory()

    def _volatile_memory(self):
        if len(self.nodes) > self._max_num_node:
            for i in range(int(self._max_num_node/2)):
                del self.nodes[i]

    def get_feature(self, index=1):
        if index > Track._max_save_feature or index > len(self.nodes):
            return None, None
        return self.nodes[-index].feature, self.id

class Tracks:
    '''
    Track set. It contains all the tracks and manage the tracks. it has the following information
    1) tracks. the set of tracks
    2) keep the previous image and features
    '''
    def __init__(self):
        self.tracks = list() # the set of tracks
        self.max_drawing_track = 10

    def __getitem__(self, item):
        return self.tracks[item]

    def append(self, track):
        self.tracks.append(track)
        self.volatile_tracks()

    def volatile_tracks(self):
        if len(self.tracks) > config['max_object']:
            # start to delete the most oldest tracks
            all_ages = [t.age for t in self.tracks]
            oldest_track_index = np.argmax(all_ages)
            del self.tracks[oldest_track_index]

    def get_track_by_id(self, id):
        for t in self.tracks:
            if t.id == id:
                return t
        return None

    def get_features(self, index = 1):
        track_id = list()
        features = list()

        # for each tracks, abstract the saved feature
        for t in self.tracks:
            feature, id = t.get_feature(index)
            # if feature is None:
            #     continue
            if feature is None:
                return None, None
            features.append(feature)
            track_id.append(id)

        if len(features) == 0:
            return None, None

        return torch.cat(features, dim=1), track_id

    def get_all_features(self):
        all_features = list()
        all_track_ids = list()
        for i in range(1, Track._max_save_feature+1):
            features, track_id = self.get_features(i)
            if features is None:
                continue

            all_features.append(features)
            all_track_ids.append([(id, i) for id in track_id])

        if len(all_features) == 0:
            return None, None
        return all_features, all_track_ids

    def one_frame_pass(self):
        keep_track_set = list()
        for i, t in enumerate(self.tracks):
            t.add_age()
            if t.age < t._max_age:
                keep_track_set.append(i)

        self.tracks = [self.tracks[i] for i in keep_track_set]


    def show(self, image):
        h, w, _ = image.shape

        # draw rectangle
        for t in self.tracks:
            if len(t.nodes) > 0 and t.age<2:
                b = t.nodes[-1].box
                image = cv2.putText(image, str(t.id), (int(b[0]*w),int((b[1])*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, t.color, 3)
                image = cv2.rectangle(image, (int(b[0]*w),int((b[1])*h)), (int((b[0]+b[2])*w), int((b[1]+b[3])*h)), t.color, 2)

        # draw line
        for t in self.tracks:
            if t.age > 1:
                continue
            if len(t.nodes) > self.max_drawing_track:
                start = len(t.nodes) - self.max_drawing_track
            else:
                start = 0
            for n1, n2 in zip(t.nodes[start:], t.nodes[start+1:]):
                c1 = (int((n1.box[0] + n1.box[2]/2.0)*w), int((n1.box[1] + n1.box[3])*h))
                c2 = (int((n2.box[0] + n2.box[2] / 2.0) * w), int((n2.box[1] + n2.box[3]) * h))
                image = cv2.line(image, c1, c2, t.color, 2)

        return image

# init_tracker_config()

# The tracker is compatible with pytorch (cuda)
class SSTTracker:
    def __init__(self,
                 model_path = config['resume'],
                 cuda = config['cuda'],
                 image_size = config['sst_dim'],
                 mean_pixel = config['mean_pixel'],
                 max_object = config['max_object']):
        self.first_run = True
        self.image_size = (image_size, image_size)
        self.model_path = model_path
        self.cuda = cuda
        self.mean_pixel = mean_pixel
        self.max_object = max_object
        self.load_model()
        self.tracks = Tracks()

    def load_model(self):
        # load the model
        self.sst = build_sst('test', 900)
        if self.cuda:
            cudnn.benchmark = True
            self.sst.load_state_dict(torch.load(config['resume']))
            self.sst = self.sst.cuda()
        else:
            self.sst.load_state_dict(torch.load(config['resume'], map_location='cpu'))
        self.sst.eval()

    def _transform_image(self, image):
        '''
        transform image to the FloatTensor (1, 3,size, size)
        :param image: same as update parameter
        :return: the transformed image FloatTensor (i.e. 1x3x900x900)
        '''
        image = cv2.resize(image, self.image_size).astype(np.float32)
        image -= self.mean_pixel
        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        image.unsqueeze_(dim=0)
        if self.cuda:
            return Variable(image.cuda())
        return Variable(image)

    def _transform_detection(self, detection):
        '''
        get the center point of detection and transform the detection into the format of FloatTensor.
        :param detection: same as detection (ie. 10x2 numpy)
        :return: The transformed Tensor, and the mask of it, FloatTensor (i.e. 1x10x1x1x2)
        '''
        # get the center, and format it in (-1, 1)
        center = (2*detection[:, 0:2] + detection[:, 2:4]) - 1.0
        # center = torch.FloatTensor(center)
        center = torch.from_numpy(center.astype(float)).float()
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)
        # rest_num = self.max_object-center.shape[1]
        # valid = torch.cat([torch.ones(1, 1, center.shape[1]),
        #                    torch.zeros(1, 1, rest_num),
        #                    torch.ones(1, 1, 1)])
        # filled_data = torch.ones(1, rest_num, 1, 1, 2) * 1.5
        # center = torch.cat([center, filled_data], dim=1)
        if self.cuda:
            return Variable(center.cuda())
        return Variable(center)

    def iou(self, bb_test, bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return (o)

    def custom_iou(self, bb_test, bb_gt):
        xx1 = np.minimum(bb_test[0], bb_gt[0])
        yy1 = np.minimum(bb_test[1], bb_gt[1])
        xx2 = np.maximum(bb_test[2], bb_gt[2])
        yy2 = np.maximum(bb_test[3], bb_gt[3])
        w = xx2 - xx1
        h = yy2 - yy1
        return ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                  + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]))/w*h


    def convert2bbox(self, detection, w, h):
        return np.array([detection[0]*w, detection[1]*h, (detection[0]+detection[2])*w, (detection[1]+detection[3])*h])

    def get_y(self, ys, ids):
        # 1. for each track_id, get y
        res = {}
        id_num = {}
        for id, y in zip(ids, ys):
            index_id = range(len(id))
            sum_index = sum(index_id)
            weight = [(i+1) / (sum_index+1) for i in index_id]
            for i in id:
                res.setdefault(i[0], []).append(y[id.index(i), :])

        # 2. get the average
        y = list()
        keys = res.keys()
        for id in keys:
            y.append(sum(res[id]) / len(res[id]))

        return np.array(y), list(keys)


    def update(self, image, detection, show_image):
        '''
        Update the state of tracker, the following jobs should be done:
        1) extract the features
        2) stack the features together
        3) get the similarity matrix
        4) do assignment work
        5) save the previous image
        :param image: the opencv readed image, format is hxwx3
        :param detections: detection array. numpy array (l, r, w, h) and they all formated in (0, 1)
        '''

        # format the image and detection
        h, w, _ = image.shape
        image_org = image
        image = self._transform_image(image)
        detection_org = detection
        detection = self._transform_detection(detection) #1x10x1x1x2

        # features can be (1, 10, 450)
        features = self.sst.forward_feature_extracter(image, detection)
        if self.first_run:
            self.first_run = False
            for i in range(detection.shape[1]):
                t = Track()
                n = Node(detection_org[i],
                         features[:, i:i + 1, :].data,
                         detection[:, i:i + 1, :, :, :].data)
                t.add_node(n)
                self.tracks.append(t)
            self.tracks.one_frame_pass()
            return

        #1) get the previous features and the track ids
        previous_features, ids = self.tracks.get_all_features()

        #2) stack the previous features and current features together
        ys = list()
        for feature in previous_features:
            if feature.shape[0] == 0 or feature.shape[1] == 0:
                continue
            y = self.sst.forward_stacker_features(Variable(feature), features)
            ys.append(y)

        y, ids = self.get_y(ys, ids)
        #3) find the corresponding by the similar matrix
        row_index, col_index = linear_sum_assignment(-y)
        # row_index, col_index = np.array(range(y.shape[0])), np.array(np.argmax(y, axis=1))
        col_index[col_index >= detection_org.shape[0]] = -1

        #4) update the tracks
        for i in row_index:
            track_id = ids[i]
            t = self.tracks.get_track_by_id(track_id)
            col_id = col_index[i]
            if col_id < 0:
                continue
            node = Node(detection_org[col_id],
                        features[:, col_id:col_id + 1, :].data,
                        detection[:, col_id:col_id + 1, :, :, :].data)
            t.add_node(node)

        #5) add new track
        for col in range(len(detection_org)):
            if col not in col_index:
                node = Node(detection_org[col],
                            features[:, col:col+1, :].data,
                            detection[:, col:col+1, :, :, :].data)
                t = Track()
                t.add_node(node)
                self.tracks.append(t)


        self.tracks.one_frame_pass()

        if show_image:
            image_org = self.tracks.show(image_org)
            return image_org

        # image_org = cv2.resize(image_org, (320, 240))
        # vw.write(image_org)

        # plt.imshow(image_org)
