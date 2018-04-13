# SST (Single Shot Tracker)
## Purpose
Designing an end-to-end deep learning tracker.
## Task
|Title|Start|Due|Detail|
|---|---|---|---|
Test KITTI | 2018/04/18 | 2018/04/20 | Get the result of KITTI and upload the result to the official website.
Continue Train KITTI | 2018/04/16 | 2018/04/18 | Continue training KITTI
Training KITTI dataset | 2018/04/11 | 2018/04/16 | Training KITTI dataset
Complete ReadMe | 2018/04/13 | 2018/04/14 | Complete the basic content of ReadMe



## Requrement
Our network is designed by pytorch framework. It is also trained and tested in Ubuntu13.1 system. The following package is required.
- python package
    - cuda8
    - python version 3.5
    - numpy
    - pandas
    - [pytorch0.31]( http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl )
    - torchvision
    - python-opencv
    - [motmetrics](https://github.com/cheind/py-motmetrics)
    - [tensorboardX](https://github.com/lanpa/tensorboard-pytorch.git)

# Result
## 2018/04/13 Training KITTI
we trained the kitti dataset by the following parameters

|parameter name| value |
|---|---|
|learning rate| 0-50k(1e-2), 50k-65k(1e-3)|
|maximum gap| 30 |

The result and accuracy is shown below:
![](image/training20180410.png)
![](image/accuracy20180410.png)
> we find that if we decrease the learning or keep it as 1e-3, we get get better result.

# Train
We provide a convient way to train the network. Currently, MOT dataset, KITTI dataset is supported.
## MOT
- Go to the root folder of this project
- In config/config.py, change the configure in function init_train_mot() and decomment the code as follows:
 ```python
    init_train_kitti()
 ```
- Run the following command
```python
PYTHONPAHT=. python train_kitti.py
```

## KITTI
- Go to the root folder of this project
- In config/config.py, change the configure in function init_train_kitti() and decomment the code as follows:
 ```python
    init_train_kitti()
 ```
- Run the following command
```python
PYTHONPAHT=. python train_kitti.py
```

# Test
Similar to the train, we also provide a convient way to test the network. Currently only support MOT dataset, KITTI dataset.

## MOT
- Go to the project folder
- In config/config.py, change the configure in function init_test_mot() and decommend the code as follows:
```python
    init_test_mot()
```
- Run the following command:
```python
PYTHONPATH=. python test_mot.py
```
## KITTI
- Go to the project folder
- In config/config.py, change the configure in function init_test_kitti() and decommend the code as follows:
```python
    init_test_kitti()
```
- Run the following command:
```python
PYTHONPATH=. python test_kitti.py
```
