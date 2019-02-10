import os

def get_sequence_name_by_det_folder(det_folder):
    '''
    get the sequence name (i.e. ['MVI_39031', 'MVI_39051']) from the detection folder
    :param det_folder: currently detection folder
    :return: return sequence names
    '''
    return [f[:9] for f in os.listdir(det_folder)]
