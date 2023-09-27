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
from models.model_utils import create_model
from utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from data_process.kitti_data_utils import Calibration
from utils.demo_utils import parse_demo_configs, do_detect
from object_tracker import ObjectTracker

if __name__ == '__main__':
    configs = parse_demo_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)
    result = []
    objects = ObjectTracker()
    t = 0
    delta_t = 0
    video_time = 10
    with torch.no_grad():
        for sample_idx in range(len(demo_dataset)):
            # t1 = time.time()
            metadatas, bev_map, img_rgb = demo_dataset.load_bevmap_front(sample_idx)
            detections, bev_map, fps = do_detect(configs, model, bev_map, is_front=True)         
            
            calib = Calibration(configs.calib_path)
            kitti_dets = convert_det_to_real_values(detections)

            objects.identify_object(kitti_dets, t, delta_t)
            objects.show_objects()
            delta_t = video_time/len(demo_dataset)
            t += delta_t
            
            # Draw prediction in the image
            bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections, objects, configs.num_classes)

            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            
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
            # t2 = time.time()
            # delta_t = t2-t1
            # t += delta_t

    df = pd.DataFrame(objects.all_detections)
    df.to_csv('motion_vectors.csv', index=False, header=False)

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
