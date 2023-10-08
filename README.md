# lidar_object_tracking

## Main programs developed:

- motion_vectors.py
- object_tracker.py
- demo_motion_vectors.py

## RUN:

Inside /lidar_object_tracking/sfa/
Run "python demo_motion_vectors.py --gpu_idx 0 --peak_thresh 0.2"

## Dataset:

https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

Velodyne point clouds (29 GB)
Training labels of object data set (5 MB)
Camera calibration matrices of object data set (16 MB)
Left color images of object data set (12 GB) (For visualization purpose only)

lidar_object_tracking/
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/

