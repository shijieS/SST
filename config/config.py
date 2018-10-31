import numpy as np
import json

configure_names = ['init_test_mot16', 'init_test_mot17', 'init_train_mot17',
                   'init_train_mot15', 'init_test_mot15', 'init_test_mot15_train_dataset',
                   'init_train_kitti', 'init_test_kitti',
                   'init_train_ua', 'init_test_ua',
                   'exp_init_train_ua', 'exp_init_test_ua',
                   'exp_test_mot17_final_net', 'exp_train_mot17_final_net',
                   'init_train_mot17_final_net_lab', 'exp_test_mot17_final_net']

current_select_configure = 'init_test_mot17'

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
        '900': [1040, 512, 256, 128, 64, 1],
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

all_functions = []


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


all_functions += [init_test_mot16]


def init_test_mot17():
    '''
    ssm
    '''
    config['resume'] = '/media/ssm/seagate/weights/MOT17/0601-E120-M80-G30-weights/sst300_0712_83000.pth'
    config['mot_root'] = '/media/ssm/seagate/dataset/MOT17'
    config['log_folder'] = '/media/ssm/seagate/logs/1008-age-node'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['type'] = 'test'


all_functions += [init_test_mot17]


def init_train_mot17():
    config['epoch_size'] = 664
    config['mot_root'] = '/media/ssm/seagate/dataset/MOT17'
    config['base_net_folder'] = '/media/ssm/seagate/ssj/weights/MOT17/vgg16_reducedfc.pth'
    config['log_folder'] = '/media/ssm/seagate/weights/MOT17/0601-E120-M80-G30-log'
    config['save_folder'] = '/media/ssm/seagate/weights/MOT17/0601-E120-M80-G30-weights'
    config['save_images_folder'] = '/media/ssm/seagate/weights/MOT17/0601-E120-M80-G30-images'
    config['type'] = 'train'
    config['dataset_type'] = 'train'
    config['resume'] = None
    config['detector'] = 'DPM'
    config['start_iter'] = 0
    config['iteration_epoch_num'] = 120
    config['iterations'] = config['start_iter'] + config['epoch_size'] * config['iteration_epoch_num'] + 50
    config['batch_size'] = 8
    config['learning_rate'] = 1e-2
    config['learning_rate_decay_by_epoch'] = (50, 80, 100, 110)
    config['save_weight_every_epoch_num'] = 5
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 30
    config['false_constant'] = 10
    config['num_workers'] = 16
    config['cuda'] = True
    config['max_object'] = 80
    config['min_visibility'] = 0.3


all_functions += [init_train_mot17]


def exp_train_mot17_final_net():
    config['epoch_size'] = 664
    config['mot_root'] = '/home/ssm/ssj/dataset/MOT17'
    config['base_net_folder'] = '/home/ssm/ssj/weights/MOT17/vgg16_reducedfc.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/MOT17/0528-E120-M80-G30-log'
    config['save_folder'] = '/home/ssm/ssj/weights/MOT17/0528-E120-M80-G30-weights'
    config['save_images_folder'] = '/home/ssm/ssj/weights/MOT17/0528-E120-M80-G30-images'
    config['type'] = 'train'
    config['dataset_type'] = 'train'
    config['resume'] = None
    config['detector'] = 'FRCNN'
    config['start_iter'] = 0
    config['iteration_epoch_num'] =  120
    config['iterations'] = config['start_iter'] + config['epoch_size']*config['iteration_epoch_num'] + 50
    config['batch_size'] = 8
    config['learning_rate'] = 1e-2
    config['learning_rate_decay_by_epoch'] = (50, 80, 100, 110)
    config['save_weight_every_epoch_num'] = 5
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 30
    config['false_constant'] = 10
    config['num_workers'] = 16
    config['cuda'] = True
    config['max_object'] = 80
    config['min_visibility'] = 0.3
    config['final_net']['900'] = [int(config['final_net']['900'][0]), 1]


all_functions += [exp_train_mot17_final_net]


def exp_test_mot17_final_net():
    config['resume'] = '/media/jianliu/Data/ssj/github/SST/weights/sst300_0712_83000.pth'
    config['mot_root'] = '/media/jianliu/Data/ssj/dataset/MOT17'
    config['log_folder'] = '/media/jianliu/Data/ssj/logs/sst/log0903-2'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['type'] = 'test'


all_functions += [exp_test_mot17_final_net]


def exp_train_mot17_final_net_lab():
    config['mot_root'] = '/media/jianliu/ssm/dataset/dataset/MOT/17/MOT17'
    config['base_net_folder'] = '/media/jianliu/ssm/ssj/github/weights/vgg16_reducedfc.pth'
    config['log_folder'] = '/media/jianliu/ssm/ssj/train/0601-E120-M80-G30-log'
    config['save_folder'] = '/media/jianliu/ssm/ssj/train/0601-E120-M80-G30-weights'
    config['save_images_folder'] = '/media/jianliu/ssm/ssj/train/0601-E120-M80-G30-images'
    config['type'] = 'train'
    config['dataset_type'] = 'train'
    config['resume'] = None
    config['detector'] = 'FRCNN'
    config['start_iter'] = 0
    config['iteration_epoch_num'] = 120
    config['iterations'] = config['start_iter'] + config['epoch_size'] * config['iteration_epoch_num'] + 50
    config['batch_size'] = 8
    config['learning_rate'] = 1e-2
    config['learning_rate_decay_by_epoch'] = (50, 80, 100, 110)
    config['save_weight_every_epoch_num'] = 5
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 30
    config['false_constant'] = 10
    config['num_workers'] = 16
    config['cuda'] = True
    config['max_object'] = 80
    config['min_visibility'] = 0.3


all_functions += [exp_train_mot17_final_net_lab]


def init_train_mot15():
    config['epoch_size'] = 664
    config['mot_root'] = '/media/ssm/seagate/dataset/MOT15/2DMOT2015'
    config['base_net_folder'] = '/home/ssm/ssj/weights/MOT17/vgg16_reducedfc.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/MOT15/1004-E120-M80-G30-log'
    config['save_folder'] = '/home/ssm/ssj/weights/MOT15/1004-E120-M80-G30-weights'
    config['save_images_folder'] = '/home/ssm/ssj/weights/MOT15/1004-E120-M80-G30-images'
    config['type'] = 'train'
    config['dataset_type'] = 'train'
    config['resume'] = None
    config['video_name_list'] = ['ADL-Rundle-6',  'ADL-Rundle-8',  'ETH-Bahnhof',  'ETH-Pedcross2',  'ETH-Sunnyday',  'KITTI-13',  'KITTI-17',  'PETS09-S2L1',  'TUD-Campus',  'TUD-Stadtmitte',  'Venice-2']
    config['start_iter'] = 0
    config['iteration_epoch_num'] = 120
    config['iterations'] = config['start_iter'] + config['epoch_size'] * config['iteration_epoch_num'] + 50
    config['batch_size'] = 8
    config['learning_rate'] = 1e-2
    config['learning_rate_decay_by_epoch'] = (50, 80, 100, 110)
    config['save_weight_every_epoch_num'] = 5
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 30
    config['false_constant'] = 10
    config['num_workers'] = 16
    config['cuda'] = True
    config['max_object'] = 80
    config['min_visibility'] = 0.3
    config['final_net']['900'] = [int(config['final_net']['900'][0]), 1]

all_functions += [init_train_mot15]

def init_test_mot15():
    config['resume'] = '/media/ssm/seagate/weights/MOT17/0601-E120-M80-G30-weights/sst300_0712_83000.pth'
    config['mot_root'] = '/media/ssm/seagate/dataset/MOT15/2DMOT2015'
    config['log_folder'] = '/media/ssm/seagate/logs/1005-mot15-test-5'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config['video_name_list'] = ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli',
                                 'ETH-Linthescher', 'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']
all_functions += [init_test_mot15]


def init_test_mot15_train_dataset():
    config['resume'] = '/media/ssm/seagate/weights/MOT17/0601-E120-M80-G30-weights/sst300_0712_83000.pth'
    config['mot_root'] = '/media/ssm/seagate/dataset/MOT15/2DMOT2015'
    config['log_folder'] = '/media/ssm/seagate/logs/1005-mot15-test-train-dataset-1'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['type'] = 'train'
    config['dataset_type'] = 'train'
    config['video_name_list'] = ['ADL-Rundle-6',  'ADL-Rundle-8',  'ETH-Bahnhof',  'ETH-Pedcross2',  'ETH-Sunnyday',  'KITTI-13',
                                 'KITTI-17',  'PETS09-S2L1',  'TUD-Campus',  'TUD-Stadtmitte',  'Venice-2']
    # config['video_name_list'] = ['KITTI-13']

all_functions += [init_test_mot15_train_dataset]


def init_train_kitti():
    config['kitti_image_root'] = '/home/ssm/ssj/dataset/KITTI/tracking/image_2'
    config['kitti_detection_root'] = '/home/ssm/ssj/dataset/KITTI/tracking/tracking_label_2'
    config['base_net_folder'] = '/home/ssm/ssj/weights/KITTI/vgg16_reducedfc.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/KITTI/log0518-I60k-M80-G30-C10-Pedestrian-Resume'
    config['save_folder'] = '/home/ssm/ssj/weights/KITTI/weights0518-I60k-M80-G30-C10-Pedestrian-Resume'
    config['type'] = 'train'
    config['dataset_type'] = 'training'
    config['resume'] = '/home/ssm/ssj/weights/KITTI/weights0406-I60k-M80-G5-C10-All-Continue/ssj300_0712_140000.pth' #None
    config['start_iter'] = 0
    config['cuda'] = True
    config['batch_size'] = 8
    config['num_workers'] = 16
    config['iterations'] = 30050
    config['learning_rate'] = 1e-2
    config['false_constant'] = 10
    config['max_object'] = 80
    config['max_gap_frame'] = 30
    config['min_gap_frame'] = 0


all_functions += [init_train_kitti]


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


all_functions += [init_test_kitti]


def init_train_ua():
    config['epoch_size'] = 10430

    config['base_net_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/vgg16_reducedfc.pth'
    config['log_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-log'
    config['save_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-weight'
    config['save_images_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-images'
    config['ua_image_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/Insight-MVT_Annotation_Train'
    config[
        'ua_detection_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/gt'
    config[
        'ua_ignore_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/DETRAC-MOT-toolkit/evaluation/igrs'
    config['resume'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-weight/sst300_0712_62580.pth'
    config['start_iter'] = 62581
    config['iteration_epoch_num'] = 10
    config['iterations'] = config['start_iter'] + config['epoch_size'] * config['iteration_epoch_num'] + 50
    config['batch_size'] = 8
    config['learning_rate'] = 1e-3
    config['learning_rate_decay_by_epoch'] = (5, 7, 8, 9)
    config['save_weight_every_epoch_num'] = 1
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 15
    config['false_constant'] = 10
    config['num_workers'] = 16
    config['cuda'] = True
    config['max_object'] = 80


all_functions += [init_train_ua]


def init_test_ua():
    config['save_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/1006-E25-M80-G30-TestSet-EB-1'
    config['ua_image_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/Insight-MVT_Annotation_Test'
    config['ua_detection_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/EB'
    config['ua_ignore_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/DETRAC-MOT-toolkit/evaluation/igrs'
    config['resume'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-weight/sst300_0712_114730.pth'
    config['detector_name'] = 'EB'
    config['batch_size'] = 1
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 30
    config['false_constant'] = 10
    config['cuda'] = True
    config['max_object'] = 80
    config['type'] = 'train'
all_functions += [init_test_ua]

def exp_init_test_ua():
    config['log_folder'] = '/media/jianliu/Data/ssj/logs/sst/ua-log-0905-1'
    config['save_folder'] = '/media/jianliu/Data/ssj/logs/sst/ua-log-0905-1'
    config['ua_image_root'] = '/media/jianliu/Data/ssj/dataset/ua-detrac/Insight-MVT_Annotation_Test'
    config['ua_detection_root'] = '/media/jianliu/Data/ssj/dataset/ua-detrac/EB'
    config['ua_ignore_root'] = '/media/jianliu/Data/ssj/dataset/ua-detrac/DETRAC-MOT-toolkit/evaluation/igrs'
    config['resume'] = '/media/jianliu/Data/ssj/weights/ua/ua-sst300_0712_114730.pth'
    config['detector_name'] = 'EB'
    config['batch_size'] = 1
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 30
    config['false_constant'] = 10
    config['cuda'] = True
    config['max_object'] = 80
    config['type'] = 'train'

all_functions += [exp_init_test_ua]

for f in all_functions:
    if f.__name__ == current_select_configure:
        f()
        break

print('use configure: ', current_select_configure)
