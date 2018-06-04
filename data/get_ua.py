import wget
from google_drive_downloader import GoogleDriveDownloader as gdd
import zipfile
import os

current_dir = os.path.abspath('./')
print('all the zip files will be download in ', current_dir, '===>')

all_link = {
    'train data': 'http://detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip',
    'test data': 'http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip',
    'DPM train': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Detections/DPM-train.zip',
    'ACF train': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Detections/ACF-train.zip',
    'R-CNN train': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Detections/R-CNN-train.zip',
    'CompACT train': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Detections/CompACT-train.zip',
    'DPM test': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Test-Detections/DPM-test.zip',
    'ACF test': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Test-Detections/ACF-test.zip',
    'R-CNN test': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Test-Detections/R-CNN-test.zip',
    'CompACT test': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Test-Detections/CompACT-test.zip',
    'Train Annotations XML': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-XML.zip',
    'Train Annotations MAT': 'http://detrac-db.rit.albany.edu/Data/DETRAC-Train-Annotations-MAT.zip',
    'MOT toolkit': 'http://detrac-db.rit.albany.edu/Data/DETRAC-MOT-toolkit.zip'
}

print('start download all the UA-DETRAC dataset ======>')

for l in all_link.keys():
    print('start download ===> ', l)
    wget.download(all_link[l])

google_drive_link = {
    'EB-Train': '1KZ8ZR2NjmlKCiRCuVwaEu0yZhskbVsgU',
    'EB-Test': '13jcpL01ac8ajqLVtT4Eth9JSDs7nR5hD'
}

for l in google_drive_link.keys():
    print('start download ===>', l)
    gdd.download_file_from_google_drive(
        file_id=google_drive_link[l],
        dest_path='./' + l+'.zip',
        unzip=False
    )


print('start unzip all files ============>')


files_name = [
    'DETRAC-train-data.zip',
    'DETRAC-test-data.zip',
    'DETRAC-Train-Detections/DPM-train.zip',
    'DETRAC-Train-Detections/ACF-train.zip',
    'R-CNN-train.zip',
    'CompACT-train.zip',
    'DPM-test.zip',
    'ACF-test.zip',
    'R-CNN-test.zip',
    'CompACT-test.zip',
    'DETRAC-Train-Annotations-XML.zip',
    'DETRAC-Train-Annotations-MAT.zip'
    'DETRAC-MOT-toolkit.zip',
    'EB-Train.zip',
    'EB-Test.zip'
]

for f in files_name:
    print('start unzip ', f)
    full_name = os.path.abspath(f)  # get full path of files
    zip_ref = zipfile.ZipFile(full_name)  # create zipfile object
    zip_ref.extractall(current_dir)  # extract file to dir
    zip_ref.close()  # close file
    os.remove(full_name)  # delete zipped file