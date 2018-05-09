import numpy as np
import json

config = {
    'mot_root': r'/home/ssm/ssj/dataset/MOT17',
    'save_folder': '/home/ssm/ssj/weights/MOT17/weights0326-I50k-M80-G30',
    'log_folder': '/home/ssm/ssj/weights/MOT17/log0326-I50k-M80-G30',
    'base_net_folder': '/home/ssm/ssj/weights/MOT17/vgg16_reducedfc.pth',
    'resume': '/home/ssm/ssj/weights/MOT17/weights0317-I50k-M80-G30v1.pth', #None,
    'start_iter': 55050,
    'cuda': True,
    'batch_size': 8,
    'num_workers': 16,
    'iterations': 85050,
    'learning_rate': 5e-3,
    'false_constant': 10,
    'type': 'train', # choose from ('test', 'train')
    'dataset_type': 'train', # choose from ('test', 'train')
    'detector': 'FRCNN', # choose from ('DPM', 'FRCNN', 'SDP')
    'max_object': 80,  # N
    'max_gap_frame': 40, # not the hard gap
    'min_gap_frame': 0, # not the hard gap
    'sst_dim': 900,
    'min_visibility': 0.3,
    'mean_pixel': (104, 117, 123),
    'max_expand': 1.2,
    'lower_contrast': 0.7,
    'upper_constrast': 1.5,
    'lower_saturation': 0.7,
    'upper_saturation': 1.5,
    'alpha_valid': 0.8,
    'base_net': {
        '900': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                'C', 512, 512, 512, 'M', 512, 512, 512],
        '1024': [],},
    'extra_net': {
        '900': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256,
                128, 'S', 256, 128, 256],  # new: this line
        '1024': [],
    },
    'selector_size': (255, 113, 56, 28, 14, 12, 10, 5, 3),
    'selector_channel':(60, 80, 100, 80, 60, 50, 40, 30, 20),
    'final_net' : {
        '900': [400, 512, 256, 128, 64, 1],
        '1024': []
    },
    'vgg_source' : [15, 25, -1],
    'default_mbox': { # The default box setup
        '900': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
        '1024': [],
    }
}

# add the contraints
config['final_net']['900'][0] = np.sum(config['selector_channel'])*2

def init_train_mot():
    config['mot_root'] = '/ssm/ssj/dataset/MOT17'
    config['base_net_folder'] = '/ssm/ssj/weights/MOT17/vgg16_reducedfc.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/MOT17/log0326-I50k-M80-G30'
    config['save_folder'] = '/home/ssm/ssj/weights/MOT17/weights0326-I50k-M80-G30'
    config['type'] = 'train'
    config['dataset_type'] = 'train'
    config['resume'] = None
    config['start_iter'] = 0
    config['cuda'] = True
    config['max_gap_frame'] = 20
    config['min_gap_frame'] = 0
    config['batch_size'] = 8
    config['num_workers'] = 16
    config['iterations'] = 85050
    config['learning_rate'] = 5e-3
    config['false_constant'] = 10
    config['detector'] = 'FRCNN'
    config['max_object'] = 80
    config['max_gap_frame'] = 40
    config['min_gap_frame'] = 0
    config['min_visibility'] = 0.3

# init_mot_train()

def init_eval():
    '''
    ssm server

    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config['resume'] = '/home/ssm/ssj/weights/MOT17/weights0303-Formal-B8-M2-F10-I14k/ssj300_0712_5000.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/MOT17/eval0302-Formal-B8-M30-F10-I40k-5000'
    config['batch_size'] = 1
    config['cuda'] = True
    config['num_workers'] = 8
    config['start_iter'] = 0
    config['iterations'] = 19000
    '''

    '''
    OpenStack Server
    '''
    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config['resume'] = '/ssm/ssj/weights/MOT17/weights0303-Formal-B8-M2-F10-I20k-5000/ssj300_0712_5000.pth'
    config['mot_root'] = '/ssm/ssj/dataset/MOT17'
    config['cuda'] = False
    config['log_folder'] = '/ssm/ssj/weights/MOT17/eval0303-Formal-B8-M2-F10-I20k-5000'
    config['max_gap_frame'] = 20
    config['min_gap_frame'] = 1
    config['batch_size'] = 1
    config['num_workers'] = 8
    config['start_iter'] = 0
    config['iterations'] = 20000
    config['write_csv'] = False

    '''
    local server
    args.trained_model = r'F:\PeopleCounting\weights\MOT17\sst300_0712_20000.pth'
    args.save_folder = r'F:\PeopleCounting\weights\MOT17\eval0223'
    args.mot_root = r'F:\PeopleCounting\dataset\MOT\17\MOT17'
    config['cuda'] = False
    '''


def init_test():
    '''
    ssm server
    '''
    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config['resume'] = '/home/ssm/ssj/weights/MOT17/weights0303-Formal-FixedClone-B8-M2-F10-I20k/ssj300_0712_15000.pth'
    config['mot_root'] = '/ssm/ssj/dataset/MOT17'
    config['cuda'] = True
    config['log_folder'] = '/home/ssm/ssj/weights/MOT17/test0303-Formal-FixedClone-B8-M2-F10-I20k-15000'
    config['max_gap_frame'] = 1
    config['min_gap_frame'] = 1
    config['batch_size'] = 1
    config['num_workers'] = 8
    config['start_iter'] = 0
    config['write_csv'] = True
    config['tensorboard'] = False

    '''
    OpenStack Server
    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config['resume'] = '/ssm/ssj/weights/MOT17/weights0303-Formal-FixedClone-B8-M2-F10-I20k/ssj300_0712_15000.pth'
    config['mot_root'] = '/ssm/ssj/dataset/MOT17'
    config['cuda'] = False
    config['log_folder'] = '/ssm/ssj/weights/MOT17/debug0303-Formal-FixedClone-B8-M2-F10-I20k-15000'
    config['max_gap_frame'] = 1
    config['min_gap_frame'] = 1
    config['batch_size'] = 1
    config['num_workers'] = 8
    config['start_iter'] = 0
    config['write_csv'] = True
    config['tensorboard'] = False
    '''

    '''
    local server
    args.trained_model = r'F:\PeopleCounting\weights\MOT17\sst300_0712_20000.pth'
    args.save_folder = r'F:\PeopleCounting\weights\MOT17\eval0223'
    args.mot_root = r'F:\PeopleCounting\dataset\MOT\17\MOT17'
    config['cuda'] = False
    '''


def init_mot_metric():
    '''
    ssm server
    '''
    config['type'] = 'test'
    config['dataset_type'] = 'train'
    config['resume'] = '/home/ssm/ssj/weights/MOT17/weights0303-Formal-FixedClone-B8-M2-F10-I20k/ssj300_0712_15000.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/MOT17/mot_metric0303-Formal-FixedClone-B8-M2-F10-I20k-15000'
    config['min_visibility'] = 0.3
    config['batch_size'] = 1
    config['start_iter'] = 0
    config['max_gap_frame'] = 1
    config['min_gap_frame'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['origin_image_height'] = 1080
    config['origin_image_width'] = 1920

    '''
    OpenStack Server

    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config['resume'] = '/ssm/ssj/weights/MOT17/weights0303-Formal-FixedClone-B8-M2-F10-I20k/ssj300_0712_15000.pth'
    config['mot_root'] = '/ssm/ssj/dataset/MOT17'
    config['cuda'] = False
    config['log_folder'] = '/ssm/ssj/weights/MOT17/mot_metric0303-Formal-FixedClone-B8-M2-F10-I20k-15000'
    config['max_gap_frame'] = 1
    config['min_gap_frame'] = 1
    config['batch_size'] = 1
    config['num_workers'] = 1
    config['start_iter'] = 0
    config['write_file'] = True
    config['tensorboard'] = False
    '''
    '''
    local server
    args.trained_model = r'F:\PeopleCounting\weights\MOT17\sst300_0712_20000.pth'
    args.save_folder = r'F:\PeopleCounting\weights\MOT17\eval0223'
    args.mot_root = r'F:\PeopleCounting\dataset\MOT\17\MOT17'
    config['cuda'] = False
    '''


def init_tracker_config():
    '''
    ssm server
    '''
    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config[
        'resume'] = '/home/ssm/ssj/weights/MOT17/weights0303-Formal-FixedClone-B8-M1-20-F10-I20k-Continue/ssj300_0712_20000.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/MOT17/mot_metric0303-Formal-FixedClone-B8-M2-F10-I20k-15000'
    config['min_visibility'] = 0.3
    config['batch_size'] = 1
    config['start_iter'] = 0
    config['max_gap_frame'] = 1
    config['min_gap_frame'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['origin_image_height'] = 1080
    config['origin_image_width'] = 1920

    '''
    OpenStack Server

    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config['resume'] = '/home/ssm/ssj/weights/MOT17/weights0303-Formal-FixedClone-B8-M10-20-F10-I20k/ssj300_0712_15000.pth'
    config['mot_root'] = '/ssm/ssj/dataset/MOT17'
    config['cuda'] = False
    config['log_folder'] = '/ssm/ssj/weights/MOT17/mot_metric0303-Formal-FixedClone-B8-M2-F10-I20k-15000'
    config['max_gap_frame'] = 1
    config['min_gap_frame'] = 1
    config['batch_size'] = 1
    config['num_workers'] = 1
    config['start_iter'] = 0
    config['write_file'] = True
    config['tensorboard'] = False
    '''
    '''
    local server
    args.trained_model = r'F:\PeopleCounting\weights\MOT17\sst300_0712_20000.pth'
    args.save_folder = r'F:\PeopleCounting\weights\MOT17\eval0223'
    args.mot_root = r'F:\PeopleCounting\dataset\MOT\17\MOT17'
    config['cuda'] = False
    '''

'''
test mot train dataset
'''
def init_test_mot16():
    '''
    ssm
    '''
    config['resume'] = '/home/ssm/ssj/weights/MOT17/weights0326-I50k-M80-G30/ssj300_0712_80000.pth'
    config['mot_root'] = '/home/ssm/ssj/dataset/MOT16'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['type'] = 'test'

# init_test_mot16()


def init_test_mot17():
    '''
    ssm
    '''
    config['resume'] = '/home/ssm/ssj/weights/MOT17/weights0326-I50k-M80-G30/ssj300_0712_80000.pth'
    config['mot_root'] = '/home/ssm/ssj/dataset/MOT17'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['type'] = 'test'

init_test_mot17()

'''
train kitti dataset
'''
def init_train_kitti():
    config['kitti_image_root'] = '/home/ssm/ssj/dataset/KITTI/tracking/image_2'
    config['kitti_detection_root'] = '/home/ssm/ssj/dataset/KITTI/tracking/tracking_label_2'
    config['base_net_folder'] = '/home/ssm/ssj/weights/KITTI/vgg16_reducedfc.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/KITTI/log0406-I60k-M80-G5-C10-All-Continue'
    config['save_folder'] = '/home/ssm/ssj/weights/KITTI/weights0406-I60k-M80-G5-C10-All-Continue'
    config['type'] = 'train'
    config['dataset_type'] = 'training'
    config['resume'] = '/home/ssm/ssj/weights/KITTI/weights0406-I60k-M80-G5-C10-All-Continue/ssj300_0712_140000.pth' #None
    config['start_iter'] = 130050
    config['cuda'] = True
    config['min_gap_frame'] = 0
    config['batch_size'] = 8
    config['num_workers'] = 16
    config['iterations'] = 140050
    config['learning_rate'] = 1e-4
    config['false_constant'] = 10
    config['max_object'] = 80
    config['max_gap_frame'] = 5
    config['min_gap_frame'] = 0
# init_train_kitti()

def init_test_kitti():
    config['kitti_image_root'] = '/home/ssm/ssj/dataset/KITTI/tracking/image_2'
    config['kitti_detection_root'] = '/home/ssm/ssj/dataset/KITTI/tracking/det_2_lsvm'
    config['type'] = 'train'
    config['dataset_type'] = 'training'
    config['resume'] = '/home/ssm/ssj/weights/KITTI/weights0406-I60k-M80-G5-C10-All-Continue/ssj300_0712_140000.pth'
    config['cuda'] = True
    config['batch_size'] = 1
    config['false_constant'] = 10
    config['max_object'] = 80

# init_test_kitti()
