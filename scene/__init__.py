#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import time
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # 初始化时，未执行
        # load_iteration=True的时候，会从point_cloud/文件夹搜索对应的文件夹
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # scene_info的信息有如下：
        # point_cloud=pcd,
        # train_cameras=train_cam_infos,
        # test_cameras=test_cam_infos,
        # nerf_normalization=nerf_normalization,
        # ply_path=ply_path)

        # 判断使用的是COLMAP还是Blender，实操的时候用的都是前者
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        
        # 如果什么都没有，则填充相机COLMAP
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

         # getNerfppNorm读取所有相机的中心点位置到最远camera的距离 * 1.1
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # Store scene info and args; create camera lists lazily per-resolution on demand
        self._scene_info = scene_info
        self._resolution_scales = resolution_scales
        self._args = args
        
        # 直接读取对应的已经迭代出来的场景
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # 初始点云高斯化
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        if scale not in self.train_cameras:
            start_time = time.perf_counter()
            print(f"[Scene] Building train cameras for scale={scale}...")
            self.train_cameras[scale] = cameraList_from_camInfos(self._scene_info.train_cameras, scale, self._args)
            elapsed = time.perf_counter() - start_time
            print(f"[Scene] Finished train cameras for scale={scale} in {elapsed:.2f}s ({len(self.train_cameras[scale])} cameras)")
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        if scale not in self.test_cameras:
            start_time = time.perf_counter()
            print(f"[Scene] Building test cameras for scale={scale}...")
            self.test_cameras[scale] = cameraList_from_camInfos(self._scene_info.test_cameras, scale, self._args)
            elapsed = time.perf_counter() - start_time
            print(f"[Scene] Finished test cameras for scale={scale} in {elapsed:.2f}s ({len(self.test_cameras[scale])} cameras)")
        return self.test_cameras[scale]