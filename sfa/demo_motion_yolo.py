import argparse
import sys
import os
import time
import warnings
import zipfile

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np
import pandas as pd

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.demo_dataset import Demo_KittiDataset
from data_process.kitti_dataset import KittiDataset
from data_process.kitti_data_utils import get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap
from models.model_utils import create_model
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from data_process.kitti_data_utils import Calibration
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip
from utils.misc import make_folder
from object_tracker import ObjectTracker
from easydict import EasyDict as edict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} to control the verbosity

from ultralytics import YOLO
from collections import deque
import tensorflow as tf

from yolo_resnet.libs.bbox3d_utils import *
from yolo_resnet.yolo_train import *


####### select model  ########
select_model = 'resnet50'
# select_model ='resnet101'
# select_model = 'resnet152'
# select_model = 'vgg11'
# select_model = 'vgg16'
# select_model = 'vgg19'
# select_model = 'efficientnetb0'
# select_model = 'efficientnetb5'
# select_model = 'mobilenetv2'

bin_size = 6
input_shape = (224, 224, 3)
trained_classes = ['Car', 'Cyclist', 'Pedestrian']
# print(bbox3d_model.summary())
print('loading file ...'+select_model+'_weights.h5...!')
P2 = np.array([[718.856, 0.0, 607.1928, 45.38225], [0.0, 718.856, 185.2157, -0.1130887], [0.0, 0.0, 1.0, 0.003779761]])
dims_avg = {'Car': np.array([1.52131309, 1.64441358, 3.85728004]),
'Van': np.array([2.18560847, 1.91077601, 5.08042328]),
'Truck': np.array([3.07044968,  2.62877944, 11.17126338]),
'Pedestrian': np.array([1.75562272, 0.67027992, 0.87397566]),
'Person_sitting': np.array([1.28627907, 0.53976744, 0.96906977]),
'Cyclist': np.array([1.73456498, 0.58174006, 1.77485499]),
'Tram': np.array([3.56020305,  2.40172589, 18.60659898])}
# print(dims_avg)

# Load a 2D model
bbox2d_model = YOLO('yolov8n-seg.pt')  # load an official model
# set model parameters
bbox2d_model.overrides['conf'] = 0.9  # NMS confidence threshold
bbox2d_model.overrides['iou'] = 0.45  # NMS IoU threshold
bbox2d_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
bbox2d_model.overrides['max_det'] = 1000  # maximum number of detections per image
bbox2d_model.overrides['classes'] = 2 ## define classes
yolo_classes = ['Pedestrian', 'Cyclist', 'Car', 'motorcycle', 'airplane', 'Van', 'train', 'Truck', 'boat']

def process2D(image, track = True, device ='0'):
    bboxes = []
    if track is True:
        results = bbox2d_model.track(image, verbose=False, device=device, persist=True)

        for id_ in list(tracking_trajectories.keys()):
            if id_ not in [int(bbox.id) for predictions in results if predictions is not None for bbox in predictions.boxes if bbox.id is not None]:
                del tracking_trajectories[id_]

        for predictions in results:
            if predictions is None:
                continue

            if predictions.boxes is None or predictions.masks is None or predictions.boxes.id is None:
                continue

            for bbox, masks in zip(predictions.boxes, predictions.masks):
                ## object detections
                for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                    xmin    = bbox_coords[0]
                    ymin    = bbox_coords[1]
                    xmax    = bbox_coords[2]
                    ymax    = bbox_coords[3]
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                    bboxes.append([bbox_coords, scores, classes, id_])

                    label = (' '+f'ID: {int(id_)}'+' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                    dim, baseline = text_size[0], text_size[1]
                    cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                    cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    centroid_x = (xmin + xmax) / 2
                    centroid_y = (ymin + ymax) / 2

                    # Append centroid to tracking_points
                    if id_ is not None and int(id_) not in tracking_trajectories:
                        tracking_trajectories[int(id_)] = deque(maxlen=5)
                    if id_ is not None:
                        tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

                # Draw trajectories
                for id_, trajectory in tracking_trajectories.items():
                    for i in range(1, len(trajectory)):
                        cv2.line(image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), (int(trajectory[i][0]), int(trajectory[i][1])), (255, 255, 255), 2)
                
                ## object segmentations
                for mask in masks.xy:
                    polygon = mask
                    cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)


    if not track:
        results = bbox2d_model.predict(image, verbose=False, device=device)  # predict on an image
        for predictions in results:
            if predictions is None:
                continue  # Skip this image if YOLO fails to detect any objects
            if predictions.boxes is None or predictions.masks is None:
                continue  # Skip this image if there are no boxes or masks

            for bbox, masks in zip(predictions.boxes, predictions.masks): 
                ## object detections
                for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                    xmin    = bbox_coords[0]
                    ymin    = bbox_coords[1]
                    xmax    = bbox_coords[2]
                    ymax    = bbox_coords[3]
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                    bboxes.append([bbox_coords, scores, classes])

                    label = (' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                    dim, baseline = text_size[0], text_size[1]
                    cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                    cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                ## object segmentations
                for mask in masks.xy:
                    polygon = mask
                    cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

    return image, bboxes


def process3D(img, bboxes2d):
    DIMS = []
    bboxes = []
    for item in bboxes2d:
        bbox_coords, scores, classes, *id_ = item if len(item) == 4 else (*item, None)
        padding = 0  # Set the padding value
        xmin = max(0, bbox_coords[0] - padding)
        ymin = max(0, bbox_coords[1] - padding)
        xmax = min(img.shape[1], bbox_coords[2] + padding)
        ymax = min(img.shape[0], bbox_coords[3] + padding)


        crop = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
        patch = tf.convert_to_tensor(crop, dtype=tf.float32)
        patch /= 255.0  # Normalize to [0,1]
        patch = tf.image.resize(patch, (224, 224))  # Resize to 224x224
        patch = tf.expand_dims(patch, axis=0)  # Equivalent to reshape((1, *crop.shape))
        prediction = bbox3d_model.predict(patch, verbose = 0)

        dim = prediction[0][0]
        bin_anchor = prediction[1][0]
        bin_confidence = prediction[2][0]

        ###refinement dimension
        try:
            dim += dims_avg[str(yolo_classes[int(classes.cpu().numpy())])] + dim
            DIMS.append(dim)
        except:
            dim = DIMS[-1]

        bbox_ = [int(xmin), int(ymin), int(xmax), int(ymax)]
        theta_ray = calc_theta_ray(img, bbox_, P2)
        # update with predicted alpha, [-pi, pi]
        alpha = recover_angle(bin_anchor, bin_confidence, bin_size)
        alpha = alpha - theta_ray
        bboxes.append([bbox_, dim, alpha, theta_ray])

    return bboxes

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = 'D:\\Dev\\lidar_object_tracking'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset','kitti')
    configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
    configs.foldername = 'test_results'
    if not os.path.isdir(configs.results_dir):
        make_folder(configs.results_dir)

    return configs

if __name__ == '__main__':
    configs = parse_demo_configs()

    # Try to download the dataset for demonstration
    server_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
    download_url = '{}/{}/{}.zip'.format(server_url, configs.foldername[:-5], configs.foldername)
    download_and_unzip(configs.dataset_dir, download_url)

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    # Load the 3D model
    bbox3d_model = load_model('D:\\Dev\\lidar_object_tracking\\sfa\\yolo_resnet\\resnet50\\resnet50_weights.h5')

    # # Load the video
    # video = cv2.VideoCapture('/home/luizfpastuch/Dev/lidar_object_tracking/yolo//assets/2011_10_03_drive_0034_sync_video_trimmed.mp4')

    # ### svae results
    # # Get video information (frame width, height, frames per second)
    # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(video.get(cv2.CAP_PROP_FPS))
    # # Define the codec and create a VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
    # out = cv2.VideoWriter(select_model+'_output_video.mp4', fourcc, 15, (frame_width, frame_height))


    tracking_trajectories = {}

    out_cap = None
    selected_dataset = Demo_KittiDataset(configs)
    result = []
    objects_lidar = ObjectTracker()
    objects_camera = ObjectTracker()
    t = 0
    delta_t = 0
    video_time = 10
    with torch.no_grad():
        for sample_idx in range(len(selected_dataset)):
            print(sample_idx)
            bev_map, img_rgb, img_path = selected_dataset.load_bevmap_front(sample_idx)
            detections, bev_map, fps = do_detect(configs, model, bev_map, is_front=True)         
            # calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
#            calib = Calibration(configs.calib_path)
            kitti_dets = convert_det_to_real_values(detections)

            objects_lidar.identify_object(kitti_dets, t, delta_t)
            objects_lidar.show_objects()
            delta_t = video_time/len(selected_dataset) #time based on video
            t += delta_t

            img = img_rgb.copy() 
            img2 = img_rgb.copy()
            img3 = img_rgb.copy() 

            obj_location = []
            obj_dim = []
            obj_alpha = []
            obj_theta = []

            ## process 2D and 3D boxes
            img2D, bboxes2d = process2D(img2, track=True)
            if len(bboxes2d) > 0:
                bboxes3D = process3D(img, bboxes2d)
                if len(bboxes3D) > 0:
                    for bbox_, dim, alpha, theta_ray in bboxes3D:
                        location = plot3d(img3, P2, bbox_, dim, alpha, theta_ray)
                        obj_location.append(location)
                        obj_dim.append(dim)
                        obj_alpha.append(alpha)
                        obj_theta.append(theta_ray)
                        
            
            # Draw prediction in the image
            bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections, objects_lidar, configs.num_classes)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

#            img_bev_h, img_bev_w = bev_map .shape[:2]
#            ratio_bev = configs.output_width / img_bev_w
#            output_bev_h = int(ratio_bev * img_bev_h)
#
#            out_img = cv2.resize(bev_map , (configs.output_width, output_bev_h))

            img_bgr = cv2.cvtColor(img2D, cv2.COLOR_RGB2BGR)
            calib = Calibration(configs.calib_path)

            #if len(kitti_dets) > 0:
                #kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                #img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

            out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)

            if out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_path = os.path.join(configs.results_dir, '{}_front.avi'.format(configs.foldername))
                print('Create video writer at {}'.format(out_path))
                out_cap = cv2.VideoWriter(out_path, fourcc, 30, (out_cap_w, out_cap_h))

            out_cap.write(bev_map)

    df = pd.DataFrame(objects_lidar.all_detections)
    df.to_csv('motion_vectors.csv', index=False, header=False)

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()