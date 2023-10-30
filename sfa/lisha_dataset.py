#!/usr/bin/env python3
import time, requests, json
import numpy as np
import struct
import open3d as o3d
import base64

class DatasetLoader:

    def __init__(self):
        self.get_url ='https://iot.ufsc.br/api/get.php'
        self.lidar_query = {
            'series' : {
                'version' : '1.2',
                'unit'    : 50462721, 
                't0'      : 1533906378366683,
                'tf'      : 1533906416788899,
                'dev'   : 0,
                'id' : 1,
                'signature' : 1
            },
            'credentials' : {
                'domain' : 'a2d2',
                    'username' : 'a2d2',
                    'password' : 'hG2847#4ABlvx962'
            }
        }
        self.camera_query = {
            'series' : {
                'version' : '1.2',
                'unit'    : 35913730, 
                't0'      : 1533906414544846,
                'tf'      : 1533906415622174,
                'dev'   : 0,
                'id' : 1,
                'signature' : 1
            },
            'credentials' : {
                'domain' : 'a2d2',
                    'username' : 'a2d2',
                    'password' : 'hG2847#4ABlvx962'
            }
        }


    def convert_bin_to_pcd(self, binFilePath):
        size_float = 4
        list_pcd = []
        with open(binFilePath, "rb") as f:
            byte = f.read(size_float * 4)
            while byte:
                x, y, z, intensity = struct.unpack("ffff", byte)
                list_pcd.append([x, y, z])
                byte = f.read(size_float * 4)
        np_pcd = np.asarray(list_pcd)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pcd)
        return pcd
    
    def open_pointcloud(self, binFilePath):
        pcd = self.convert_bin_to_pcd(binFilePath)
        o3d.visualization.draw_geometries([pcd])

    def load_pointcloud(self):
        session = requests.Session()
        session.headers = {'Content-type' : 'application/json'}
        response = session.post(self.get_url, json.dumps(self.lidar_query))
        if response.status_code == 200:
            json_data = response.json()
            for i in range(len(json_data['series'])):
                decoded = base64.b64decode(base64.b64decode(json_data['series'][i]['value']))
                file_path = "lisha_dataset/lidar/pointCloud_"+str(i)+".bin"
                with open(file_path, "wb") as fh:
                    fh.write(decoded)
                
    def load_image(self):
        session = requests.Session()
        session.headers = {'Content-type' : 'application/json'}
        response = session.post(self.get_url, json.dumps(self.camera_query))
        if response.status_code == 200:
            json_data = response.json()
            for i in range(len(json_data['series'])):
                decoded = base64.b64decode(base64.b64decode(json_data['series'][i]['value']))
                file_path = "lisha_dataset/camera/image_"+str(i)+".png"
                with open(file_path, "wb") as fh:
                    fh.write(decoded)

if __name__ == '__main__':
    data_loader = DatasetLoader()
    data_loader.load_pointcloud()
    data_loader.load_image()
    data_loader.open_pointcloud("lisha_dataset/lidar/pointCloud_38.bin")