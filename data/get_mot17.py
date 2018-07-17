import wget
from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile
import os

current_dir = os.path.abspath('./')
print('all the zip files will be download in ', current_dir, '===>')

print('start download all the MOT17 dataset ======>')


all_link = {
    'MOT17': 'https://motchallenge.net/data/MOT17.zip',
}

for l in all_link.keys():
    print('start download ===> ', l)
    wget.download(all_link[l])

print('start unzip all files ============>')

files_name = [
    'MOT17.zip'
]

for f in files_name:
    print('start unzip ', f)
    full_name = os.path.abspath(f)  # get full path of files
    zip_ref = zipfile.ZipFile(full_name)  # create zipfile object
    zip_ref.extractall(current_dir)  # extract file to dir
    zip_ref.close()  # close file
    os.remove(full_name)  # delete zip file

