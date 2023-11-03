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
from utils.demo_utils import parse_demo_configs, do_detect
from utils.misc import make_folder
from object_tracker import ObjectTracker
from easydict import EasyDict as edict

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
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')
    configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
    configs.foldername = 'test_results'
    if not os.path.isdir(configs.results_dir):
        make_folder(configs.results_dir)

    return configs

if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    out_cap = None
    selected_dataset = KittiDataset(configs)
    result = []
    objects = ObjectTracker()
    t = 0
    delta_t = 0
    video_time = 10
    with torch.no_grad():
        for sample_idx in range(len(selected_dataset)):
            print(sample_idx)
            bev_map, img_rgb, img_path = selected_dataset.load_bevmap_front(sample_idx)
            detections, bev_map, fps = do_detect(configs, model, bev_map, is_front=True)         
            calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
#            calib = Calibration(configs.calib_path)
            kitti_dets = convert_det_to_real_values(detections)

            objects.identify_object(kitti_dets, t, delta_t)
            objects.show_objects()
            delta_t = video_time/len(selected_dataset) #time based on video
            t += delta_t
            
            # Draw prediction in the image
            bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections, objects, configs.num_classes)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            img_bev_h, img_bev_w = bev_map .shape[:2]
            ratio_bev = configs.output_width / img_bev_w
            output_bev_h = int(ratio_bev * img_bev_h)

            out_img = cv2.resize(bev_map , (configs.output_width, output_bev_h))
            
            if out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_path = os.path.join(configs.results_dir, '{}_front.avi'.format(configs.foldername))
                print('Create video writer at {}'.format(out_path))
                out_cap = cv2.VideoWriter(out_path, fourcc, 30, (out_cap_w, out_cap_h))

            out_cap.write(bev_map)

    df = pd.DataFrame(objects.all_detections)
    df.to_csv('motion_vectors.csv', index=False, header=False)

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
