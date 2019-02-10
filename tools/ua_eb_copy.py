'''
author: shijieSun
email: shijieSun@chd.edu.cn
'''
import argparse
import os
import numpy as np

print('''
Usage: check_ua_result --ua_root="ua root path" --template_root="your template folder such as GOG/DPM"
''')

parser = argparse.ArgumentParser(description='the tools for copy eb files to the specified folders according the sequences list')
parser.add_argument('--eb_root', default=r"D:\ssj\DETRAC\20170721Result\0803-E25-M80-G30-TestSet-EB-UA\EB-Test\EB", help='the eb folder')
parser.add_argument('--eb_destination', default=r"D:\ssj\DETRAC\20170721Result\0803-E25-M80-G30-TestSet-EB-UA\EB-Test\EBNew", help='the destination folders')
parser.add_argument('--eb_sequences', default="../config/ua_experienced.txt", help='the destination folders')


args = parser.parse_args()

print('''
basic checking
''')
if not os.path.exists(args.eb_root) or not os.path.exists(args.eb_sequences):
    raise FileNotFoundError()

if not os.path.exists(args.eb_destination):
    print('create new folder ' + args.eb_destination)
    os.mkdir(args.eb_destination)

print('''
reading eb_sequences
''')

sequences_list = np.loadtxt(args.eb_sequences, dtype='str')

print('''
checking eb folder
''')

eb_txt_name_list = [f for f in os.listdir(args.eb_root) if os.path.isfile(os.path.join(args.eb_root, f)) and os.path.splitext(f)[1]=='.txt']

txt_name_format = "{}_Det_EB.txt"
eb_selected_txt_name_list = []
for seq in sequences_list:
    eb_txt_name = txt_name_format.format(seq)
    if eb_txt_name not in eb_txt_name_list:
        raise FileNotFoundError("cannot find " + eb_txt_name)
    eb_selected_txt_name_list += [eb_txt_name]

print('''
read and copy files
''')

for txt_name in eb_selected_txt_name_list:
    print('processing '+txt_name)
    txt_path = os.path.join(args.eb_root, txt_name)
    dest_txt_path = os.path.join(args.eb_destination, txt_name)
    data = np.loadtxt(txt_path, dtype=float, delimiter=',')
    np.savetxt(dest_txt_path, data, fmt='%d,%d,%1.2f,%1.2f,%1.2f,%1.2f,%1.2f',delimiter=',')


print("finished :)")

