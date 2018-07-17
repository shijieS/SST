import os
import os.path
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
from config.config import config
import random


class Node:
    def __init__(self, box, frame_id, next_fram_id=-1):
        self.box = box
        self.frame_id = frame_id
        self.next_frame_id = next_fram_id


class Track:
    def __init__(self, id):
        self.nodes = list()
        self.id = id

    def add_node(self, n):
        if len(self.nodes) > 0:
            self.nodes[-1].next_frame_id = n.frame_id
        self.nodes.append(n)

    def get_node_by_index(self, index):
        return self.nodes[index]


class Tracks:
    def __init__(self):
        self.tracks = list()

    def add_node(self, node, id):
        node_add = False
        track_index = 0
        node_index = 0
        for t in self.tracks:
            if t.id == id:
                t.add_node(node)
                node_add = True
                track_index = self.tracks.index(t)
                node_index = t.nodes.index(node)
                break
        if not node_add:
            t = Track(id)
            t.add_node(node)
            self.tracks.append(t)
            track_index = self.tracks.index(t)
            node_index = t.nodes.index(node)

        return track_index, node_index

    def get_track_by_index(self, index):
        return self.tracks[index]


class GTSingleParser:
    def __init__(self, image_folder, detection_file_name, ignore_file_name,
                 min_gap=config['min_gap_frame'],
                 max_gap=config['max_gap_frame']):
        datatype = {0: int, 1: int, 2: float, 3: float, 4: float, 5: float}
        datatype_ignore = {0: float, 1: float, 2: float, 3: float}

        self.min_gap = min_gap
        self.max_gap = max_gap

        # 1. get the gt path and image folder
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.ignore_file_name = ignore_file_name

        # 2. read the gt data
        self.image_format = os.path.join(self.image_folder, 'img{0:05d}.jpg')
        self.detection = pd.read_csv(self.detection_file_name, sep=',', header=None, dtype=datatype)

        # 3. filter detection by the ignore region
        mask = None
        if ignore_file_name is not None and os.stat(self.ignore_file_name).st_size > 0:
            self.ignore = pd.read_csv(self.ignore_file_name, sep=',', header=None, dtype=datatype_ignore)
            self.ignore = self.ignore.values
            ls = self.detection.iloc[:, 2].values
            ts = self.detection.iloc[:, 3].values
            rs = self.detection.iloc[:, 4].values
            bs = self.detection.iloc[:, 5].values

            self.ignore = np.array([[r[0], r[1], r[0]+r[2], r[1]+r[3]] for r in self.ignore])
            for rect in self.ignore:
                l = rect[0]
                t = rect[1]
                r = rect[2]
                b = rect[3]
                res = np.logical_and(
                    np.logical_and(np.logical_and(l < ls, ls < r), np.logical_and(l < rs, rs < r)),
                    np.logical_and(np.logical_and(t < ts, ts < b), np.logical_and(t < bs, bs < b))
                )
                if mask is None:
                    mask = res
                else:
                    mask = np.logical_or(mask, res)
        if mask is not None:
            self.detection = self.detection[np.logical_not(mask)]

        self.tracks = Tracks()
        self.recorder = {}

        if len(self.detection) == 0:
            self.max_frame_index = 0
            return

        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())
        if len(self.detection_group_keys) == 0:
            self.max_frame_index = 0
        else:
            self.max_frame_index = max(self.detection_group_keys)

        # 3. update tracks
        for key in self.detection_group_keys:
            det = self.detection_group.get_group(key).values
            ids = np.array(det[:, 1]).astype(int)
            det = np.array(det[:, 2:6])

            self.recorder[key] = list()
            # 3.1 update tracks
            for id, d in zip(ids, det):
                node = Node(d, key)
                track_index, node_index = self.tracks.add_node(node, id)
                self.recorder[key].append((track_index, node_index))

    def _getimage(self, frame_index):
        image_path = self.image_format.format(frame_index)
        return cv2.imread(image_path)

    def get_item(self, frame_index):
        '''
        get the current_image, current_boxes, next_image, next_boxes, labels from the frame_index
        :param frame_index:
        :return: current_image, current_boxes, next_image, next_boxes, labels
        '''
        if frame_index not in self.recorder:
            return None, None, None, None, None

        # get current_image, current_box, next_image, next_box and labels
        current_image = self._getimage(frame_index)
        current_boxes = list()
        current = self.recorder[frame_index]
        next_frame_indexes = list()
        current_track_indexes = list()
        # 1. get current box
        for track_index, node_index in current:
            t = self.tracks.get_track_by_index(track_index)
            n = t.get_node_by_index(node_index)
            current_boxes.append(n.box)

            current_track_indexes.append(track_index)
            if n.next_frame_id != -1:
                next_frame_indexes.append(n.next_frame_id)

        # 2. decide the next frame (0.5 probability to choose the farest ones, and other probability to choose the frame between them)
        if len(next_frame_indexes) == 0:
            return None, None, None, None, None
        if len(next_frame_indexes) == 1:
            next_frame_index = next_frame_indexes[0]
        else:
            max_next_frame_index = max(next_frame_indexes)
            is_choose_farest = bool(random.getrandbits(1))
            if is_choose_farest:
                next_frame_index = max_next_frame_index
            else:
                next_frame_index = random.choice(next_frame_indexes)
                gap_frame = random.randint(self.min_gap, self.max_gap)
                temp_frame_index = next_frame_index + gap_frame
                choice_gap = list(range(self.min_gap, self.max_gap))
                if self.min_gap != 0:
                    choice_gap.append(0)
                while not temp_frame_index in self.recorder:
                    gap_frame = random.choice(choice_gap)
                    temp_frame_index = next_frame_index + gap_frame
                next_frame_index = temp_frame_index

        # 3. get next image
        next_image = self._getimage(next_frame_index)

        # 4. get next frame boxes
        next = self.recorder[next_frame_index]
        next_boxes = list()
        next_track_indexes = list()
        for track_index, node_index in next:
            t = self.tracks.get_track_by_index(track_index)
            next_track_indexes.append(track_index)
            n = t.get_node_by_index(node_index)
            next_boxes.append(n.box)

        # 5. get the labels
        current_track_indexes = np.array(current_track_indexes)
        next_track_indexes = np.array(next_track_indexes)
        labels = np.repeat(np.expand_dims(np.array(current_track_indexes), axis=1), len(next_track_indexes),
                           axis=1) == np.repeat(np.expand_dims(np.array(next_track_indexes), axis=0),
                                                len(current_track_indexes), axis=0)

        # 6. return all values
        # 6.1 change boxes format
        current_boxes = np.array(current_boxes)
        next_boxes = np.array(next_boxes)
        # 6.2 return the corresponding values
        return current_image, current_boxes, next_image, next_boxes, labels

    def __len__(self):
        return self.max_frame_index


class GTParser:
    def __init__(self, ua_image_root=config['ua_image_root'],
                 ua_detection_root=config['ua_detection_root'],
                 ignore_root=config['ua_ignore_root']
                 ):
        # analsis all the folder in mot_root
        # 1. get all the folders
        image_root = ua_image_root
        all_image_folders = sorted(
            [os.path.join(image_root, d) for d in os.listdir(image_root)]
        )

        detection_root = ua_detection_root
        all_detection_files = sorted(
            [os.path.join(detection_root, f) for f in os.listdir(detection_root)]
        )

        all_ignore_files = sorted(
            [os.path.join(ignore_root, f) for f in os.listdir(ignore_root)]
        )

        ignore_file_base_name = [os.path.basename(f)[:-8] for f in all_ignore_files]

        # 2. create single parser
        self.parsers = []
        for image_folder, detection_folder in zip(all_image_folders, all_detection_files):
            print('loading ', os.path.basename(detection_folder), '>>>')
            i = ignore_file_base_name.index(os.path.basename(image_folder))
            ignore_file = all_ignore_files[i]
            self.parsers += [GTSingleParser(image_folder, detection_folder, ignore_file)]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers]
        self.len = sum(self.lens)

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def __getitem__(self, item):
        if item < 0:
            return None, None, None, None, None
        # 1. find the parser
        total_len = 0
        index = 0
        current_item = item
        for l in self.lens:
            total_len += l
            if item < total_len:
                break
            else:
                index += 1
                current_item -= l

        # 2. get items
        if index >= len(self.parsers):
            return None, None, None, None, None
        return self.parsers[index].get_item(current_item)


class UATrainDataset(data.Dataset):
    '''
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    '''

    def __init__(self,
                 ua_image_root=config['ua_image_root'],
                 ua_detection_root=config['ua_detection_root'],
                 ua_ignore_root=config['ua_ignore_root'],
                 transform=None,
                 max_object=config['max_object'],
                 dataset_name='UA-DETRAC'
                 ):
        print('start loading dataset ', dataset_name, '>>>>>>>>')
        # 1. init all the variables
        self.ua_image_root = ua_image_root
        self.ua_detection_root = ua_detection_root
        self.ua_ignore_root = ua_ignore_root
        self.transform = transform
        self.max_object = max_object
        self.dataset_name = dataset_name

        self.frame_gap_range = np.array([[i, -i] for i in range(1, 30)]).flatten()

        # 2. init GTParser
        self.parser = GTParser(self.ua_image_root, self.ua_detection_root, self.ua_ignore_root)

        print('have loaded {} items, well done'.format(len(self)), dataset_name, '<<<<<<<<')

    def __getitem__(self, item):
        current_image, current_box, next_image, next_box, labels = self.parser[item]

        if current_image is None or labels is None:
            current_image, current_box, next_image, next_box, labels = self.parser[
                item + random.randint(-config['max_gap_frame'], config['max_gap_frame'])
            ]
            print('None processing.')

        if self.transform is None:
            return current_image, current_box, next_image, next_box, labels

        # change the label to max_object x max_object
        while labels is None or current_image is None or next_image is None:
            current_image, current_box, next_image, next_box, labels = self.parser[
                random.randint(0, len(self))
                ]
            print('None processing.')

        labels = np.pad(labels,
                        [(0, self.max_object - labels.shape[0]),
                         (0, self.max_object - labels.shape[1])],
                        mode='constant',
                        constant_values=0)
        return self.transform(current_image, next_image, current_box, next_box, labels)

    def __len__(self):
        return len(self.parser)


def test_dataset():
    # 1. test init function
    dataset = UATrainDataset()
    print(len(dataset))

    # 2. test get item
    l = len(dataset)
    for i in range(l):
        print(i)
        current_image, current_boxes, next_image, next_boxes, labels = dataset[i]
        if current_image is None:
            continue
        for i, b1 in enumerate(current_boxes):
            color = (0, 0, 255)
            if sum(labels[i, :]) == 0:
                b2 = np.array([0, 0, 0, 0])
            else:
                b2 = next_boxes[labels[i, :]][0]
                color = tuple((np.random.rand(3) * 255).astype(int).tolist())
            if sum(labels[i, :]) > 1:
                raise EnvironmentError('label error')
            b1 = b1.astype(int)
            b2 = b2.astype(int)
            cv2.rectangle(current_image, tuple(b1[:2]), tuple(b1[2:]), color, 2)
            cv2.rectangle(next_image, tuple(b2[:2]), tuple(b2[2:]), color, 2)
        image = np.concatenate([current_image, next_image], axis=0)
        h, w, _ = image.shape
        image = cv2.resize(image, (w//2, h//2))
        cv2.imshow('res', image)
        cv2.waitKey(0)

# test_dataset()
