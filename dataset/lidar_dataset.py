import copy
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from natsort import natsorted
from numpy.linalg import inv
from torch.utils.data import Dataset

from model.feature_octree import FeatureOctree
from utils.config import SHINEConfig
from utils.contours import find_contours
from utils.data_sampler import dataSampler
from utils.labels import id2label
from utils.pose import *
from utils.semantic_kitti_utils import *


class LiDARDataset(Dataset):
    def __init__(self, config: SHINEConfig, octree: FeatureOctree = None, use_outliers=False) -> None:
        super().__init__()

        self.kitti360 = config.kitti360
        self.kt360_id2kittiId = {id: label.kittiId for id, label in id2label.items()}
        self.config = config
        self.itlp_campus = config.itlp_campus
        self.dtype = config.dtype
        torch.set_default_dtype(self.dtype)
        self.device = config.device
        self.use_outliers = use_outliers

        self.calib = {}
        if config.calib_path != "":
            self.calib = read_calib_file(config.calib_path)
        else:
            self.calib["Tr"] = np.eye(4)
        if config.pose_path.endswith("txt"):
            if self.kitti360:
                self.poses_w = read_kitti360_poses_file(config.pose_path, self.calib)
            else:
                self.poses_w = read_poses_file(config.pose_path, self.calib)
        elif config.pose_path.endswith("csv"):
            self.poses_w = csv_odom_to_transforms(config.pose_path)
        else:
            sys.exit(
                "Wrong pose file format. Please use either *.txt (KITTI format) or *.csv (xyz+quat format)"
            )

        # pose in the reference frame (might be the first frame used)
        self.poses_ref = self.poses_w  # initialize size

        # point cloud files
        self.pc_filenames = natsorted(
            os.listdir(config.pc_path)
        )  # sort files as 1, 2,… 9, 10 not 1, 10, 100 with natsort
        self.total_pc_count = len(self.pc_filenames)

        # feature octree
        self.octree = octree

        self.last_relative_tran = np.eye(4)

        # initialize the data sampler
        self.sampler = dataSampler(config)
        self.ray_sample_count = config.surface_sample_n + config.free_sample_n

        # merged downsampled point cloud
        self.map_down_pc = o3d.geometry.PointCloud()
        # map bounding box in the world coordinate system
        self.map_bbx = o3d.geometry.AxisAlignedBoundingBox()
        self.cur_bbx = o3d.geometry.AxisAlignedBoundingBox()

        # get the pose in the reference frame
        self.used_pc_count = 0
        begin_flag = False
        self.begin_pose_inv = np.eye(4)
        for frame_id in range(self.total_pc_count):
            if (
                frame_id < config.begin_frame
                or frame_id > config.end_frame
                or frame_id % config.every_frame != 0
            ):
                continue
            if self.poses_w[frame_id] is None:
                continue
            if not begin_flag:  # the first frame used
                begin_flag = True
                if config.first_frame_ref:
                    self.begin_pose_inv = inv(self.poses_w[frame_id])  # T_rw
                else:
                    # just a random number to avoid octree boudnary marching cubes problems on synthetic dataset such as MaiCity(TO FIX)
                    self.begin_pose_inv[2, 3] += config.global_shift_default
            # use the first frame as the reference (identity)
            self.poses_ref[frame_id] = np.matmul(self.begin_pose_inv, self.poses_w[frame_id])
            self.used_pc_count += 1
        # or we directly use the world frame as reference

        # to cope with the gpu memory issue (use cpu memory for the data pool, a bit slower for moving between cpu and gpu)
        if (
            self.used_pc_count > config.pc_count_gpu_limit
            and not config.continual_learning_reg
            and not config.window_replay_on
        ):
            self.pool_device = "cpu"
            self.to_cpu = True
            self.sampler.dev = "cpu"
            print("too many scans, use cpu memory")
        else:
            self.pool_device = config.device
            self.to_cpu = False

        # data pool
        self.coord_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.sdf_label_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.normal_label_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.color_label_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.sem_label_pool = torch.empty((0), device=self.pool_device, dtype=torch.long)
        self.weight_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.sample_depth_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.ray_depth_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)
        self.origin_pool = torch.empty((0, 3), device=self.pool_device, dtype=self.dtype)
        self.time_pool = torch.empty((0), device=self.pool_device, dtype=self.dtype)

    def process_frame(self, frame_id, incremental_on=False, rand=False):
        pc_radius = self.config.pc_radius
        min_z = self.config.min_z
        max_z = self.config.max_z
        normal_radius_m = self.config.normal_radius_m
        normal_max_nn = self.config.normal_max_nn
        rand_down_r = self.config.rand_down_r

        if self.config.with_contour:
            rand_down_r -= 0.07  # approx value for contour points
        vox_down_m = self.config.vox_down_m
        sor_nn = self.config.sor_nn
        sor_std = self.config.sor_std

        self.cur_pose_ref = self.poses_ref[frame_id]
        if self.cur_pose_ref is None:
            return None

        # load point cloud (support *pcd, *ply and kitti *bin format)
        frame_filename = os.path.join(self.config.pc_path, self.pc_filenames[frame_id])

        if not self.config.semantic_on:
            frame_pc = self.read_point_cloud(frame_filename)
        else:
            label_filename = os.path.join(
                self.config.label_path, self.pc_filenames[frame_id].replace("bin", "label")
            )
            if self.config.with_contour:
                frame_pc, frame_labels = self.read_semantic_point_label(
                    frame_filename, label_filename, with_labels=True
                )
            else:
                frame_pc = self.read_semantic_point_label(frame_filename, label_filename)
            if frame_pc is None:
                return None

        if self.config.add_distance_noise:
            frame_pc = self.add_distance_noise(frame_pc)

        # block filter: crop the point clouds into a cube
        bbx_min = np.array([-pc_radius, -pc_radius, min_z])
        bbx_max = np.array([pc_radius, pc_radius, max_z])
        bbx = o3d.geometry.AxisAlignedBoundingBox(bbx_min, bbx_max)
        frame_pc = frame_pc.crop(bbx)

        # surface normal estimation
        if self.config.estimate_normal:
            frame_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius_m, max_nn=normal_max_nn
                )
            )

        # point cloud downsampling
        if self.config.rand_downsample or rand:
            if self.config.rand_down_way == "random":
                # random downsampling
                frame_pc = frame_pc.random_down_sample(sampling_ratio=rand_down_r)
            elif self.config.rand_down_way == "uniform":
                frame_pc = frame_pc.uniform_down_sample(every_k_points=int(1.0 / rand_down_r))
            elif self.config.rand_down_way == "farthest":
                frame_pc = frame_pc.farthest_point_down_sample(
                    num_samples=int(len(frame_pc.points) * rand_down_r)
                )
            elif self.config.rand_down_way == "semantic":
                frame_pc = self.sample_points_by_labels(frame_pc, rand_down_r)
            elif self.config.rand_down_way == "inverse_semantic":
                frame_pc = self.inverse_stratified_sampling(frame_pc, rand_down_r, self.config.min_down_r)
            else:
                raise ValueError(f"rand_down_way {self.config.rand_down_way} not supported")

            if self.config.with_contour:
                contour_cloud = find_contours(
                    frame_pc, frame_labels, mode=self.config.contour_mode, delta=self.config.contour_delta
                )  # 6-8 percent of total points

                # join contour points to the point cloud
                frame_pc = self.merge_point_clouds(frame_pc, contour_cloud)

        else:
            # voxel downsampling
            frame_pc = frame_pc.voxel_down_sample(voxel_size=vox_down_m)

        # apply filter (optional)
        if self.config.filter_noise:
            frame_pc = frame_pc.remove_statistical_outlier(sor_nn, sor_std, print_progress=False)[0]

        # load the label from the color channel of frame_pc
        if self.config.semantic_on:
            frame_sem_label = np.asarray(frame_pc.colors)[:, 0] * 255.0  # from [0-1] tp [0-255]
            frame_sem_label = np.round(frame_sem_label, 0)  # to integer value
            sem_label_list = list(frame_sem_label)
            frame_sem_rgb = [sem_kitti_color_map[sem_label] for sem_label in sem_label_list]
            frame_sem_rgb = np.asarray(frame_sem_rgb, dtype=np.float64) / 255.0
            frame_pc.colors = o3d.utility.Vector3dVector(frame_sem_rgb)

        frame_origin = self.cur_pose_ref[:3, 3] * self.config.scale  # translation part
        frame_origin_torch = torch.tensor(frame_origin, dtype=self.dtype, device=self.pool_device)

        # transform to reference frame
        frame_pc = frame_pc.transform(self.cur_pose_ref)

        # make a backup for merging into the map point cloud
        frame_pc_clone = copy.deepcopy(frame_pc)
        frame_pc_clone = frame_pc_clone.voxel_down_sample(
            voxel_size=self.config.map_vox_down_m
        )  # for smaller memory cost
        self.map_down_pc += frame_pc_clone
        self.cur_frame_pc = frame_pc_clone

        self.map_bbx = self.map_down_pc.get_axis_aligned_bounding_box()
        self.cur_bbx = self.cur_frame_pc.get_axis_aligned_bounding_box()
        # and scale to [-1,1] coordinate system
        frame_pc_s = frame_pc.scale(self.config.scale, center=(0, 0, 0))

        frame_pc_s_torch = torch.tensor(
            np.asarray(frame_pc_s.points), dtype=self.dtype, device=self.pool_device
        )

        frame_normal_torch = None
        if self.config.estimate_normal:
            frame_normal_torch = torch.tensor(
                np.asarray(frame_pc_s.normals), dtype=self.dtype, device=self.pool_device
            )

        frame_label_torch = None
        if self.config.semantic_on:
            frame_label_torch = torch.tensor(frame_sem_label, dtype=self.dtype, device=self.pool_device)

        # sampling the points
        (coord, sdf_label, normal_label, sem_label, weight, sample_depth, ray_depth) = self.sampler.sample(
            frame_pc_s_torch, frame_origin_torch, frame_normal_torch, frame_label_torch
        )

        origin_repeat = frame_origin_torch.repeat(coord.shape[0], 1)
        time_repeat = torch.tensor(frame_id, dtype=self.dtype, device=self.pool_device).repeat(coord.shape[0])

        # update feature octree
        if self.octree is not None:
            if self.config.octree_from_surface_samples:
                # update with the sampled surface points
                self.octree.update(coord[weight > 0, :].to(self.device), incremental_on)
            else:
                # update with the original points
                self.octree.update(frame_pc_s_torch.to(self.device), incremental_on)

        # get the data pool ready for training

        # ray-wise samples order
        if incremental_on:  # for the incremental mapping with feature update regularization
            self.coord_pool = coord
            self.sdf_label_pool = sdf_label
            self.normal_label_pool = normal_label
            self.sem_label_pool = sem_label
            self.weight_pool = weight
            self.sample_depth_pool = sample_depth
            self.ray_depth_pool = ray_depth
            self.origin_pool = origin_repeat
            self.time_pool = time_repeat

        else:  # batch processing
            # using a sliding window for the data pool
            if self.config.window_replay_on:
                pool_relative_dist = (self.coord_pool - frame_origin_torch).norm(2, dim=-1)
                filter_mask = pool_relative_dist < self.config.window_radius * self.config.scale

                # and also have two filter mask options (delta frame, distance)
                self.coord_pool = self.coord_pool[filter_mask]
                self.weight_pool = self.weight_pool[filter_mask]

                self.sdf_label_pool = self.sdf_label_pool[filter_mask]
                self.origin_pool = self.origin_pool[filter_mask]
                self.time_pool = self.time_pool[filter_mask]

                if normal_label is not None:
                    self.normal_label_pool = self.normal_label_pool[filter_mask]
                if sem_label is not None:
                    self.sem_label_pool = self.sem_label_pool[filter_mask]

            # or we will simply use all the previous samples

            # concat with current observations
            self.coord_pool = torch.cat((self.coord_pool, coord.to(self.pool_device)), 0)
            self.weight_pool = torch.cat((self.weight_pool, weight.to(self.pool_device)), 0)
            if self.config.ray_loss:
                self.sample_depth_pool = torch.cat(
                    (self.sample_depth_pool, sample_depth.to(self.pool_device)), 0
                )
                self.ray_depth_pool = torch.cat((self.ray_depth_pool, ray_depth.to(self.pool_device)), 0)
            else:
                self.sdf_label_pool = torch.cat((self.sdf_label_pool, sdf_label.to(self.pool_device)), 0)
                self.origin_pool = torch.cat((self.origin_pool, origin_repeat.to(self.pool_device)), 0)
                self.time_pool = torch.cat((self.time_pool, time_repeat.to(self.pool_device)), 0)

            if normal_label is not None:
                self.normal_label_pool = torch.cat(
                    (self.normal_label_pool, normal_label.to(self.pool_device)), 0
                )
            else:
                self.normal_label_pool = None

            if sem_label is not None:
                self.sem_label_pool = torch.cat((self.sem_label_pool, sem_label.to(self.pool_device)), 0)
            else:
                self.sem_label_pool = None

    def merge_point_clouds(self, cloud1, cloud2):
        """Merges two Open3D point clouds into one.

        Parameters
        ----------
        - cloud1: An Open3D point cloud object.
        - cloud2: An Open3D point cloud object.

        Returns
        -------
        - merged_cloud: A new Open3D point cloud object containing points from both input clouds.

        """
        if len(cloud2.points) == 0:
            return cloud1
        # Create a new point cloud that is a copy of the first cloud
        merged_cloud = o3d.geometry.PointCloud()
        merged_cloud.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(cloud1.points), np.asarray(cloud2.points)))
        )

        # Check if both clouds have colors and merge them
        if cloud1.colors and cloud2.colors:
            merged_cloud.colors = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(cloud1.colors), np.asarray(cloud2.colors)))
            )

        # Check if both clouds have normals and merge them
        if cloud1.normals and cloud2.normals:
            merged_cloud.normals = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(cloud1.normals), np.asarray(cloud2.normals)))
            )

        return merged_cloud

    def sample_points_by_labels(self, point_cloud, downsample_rate, uniform=False):
        # Convert Open3D PointCloud to numpy arrays
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)  # Here colors are used as semantic labels

        # Assuming the label is the first element in the color vector since they are repeated
        labels = colors[:, 0]

        # Find unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Calculate how many points to sample from each label
        num_samples_per_label = (counts * downsample_rate).astype(int)

        # Initialize lists to collect sampled points and colors
        sampled_points = []
        sampled_colors = []

        # Sample points from each label group
        for label, num_samples in zip(unique_labels, num_samples_per_label):
            # Get indices where the current label occurs
            indices = np.where(labels == label)[0]

            # Randomly choose indices based on the number of samples
            sampled_indices = np.random.choice(indices, num_samples, replace=False)

            # Append sampled points and colors
            sampled_points.append(points[sampled_indices])
            sampled_colors.append(colors[sampled_indices])

        # Concatenate all sampled points and colors
        sampled_points = np.vstack(sampled_points)
        sampled_colors = np.vstack(sampled_colors)

        # Create a new Open3D point cloud with sampled data
        sampled_point_cloud = o3d.geometry.PointCloud()
        sampled_point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_point_cloud.colors = o3d.utility.Vector3dVector(sampled_colors)

        return sampled_point_cloud

    def inverse_stratified_sampling(self, point_cloud, downsample_rate, min_rate, uniform=False):
        # Convert Open3D PointCloud to numpy arrays
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)  # Here colors are used as semantic labels

        # Assuming the label is the first element in the color vector since they are repeated
        labels = colors[:, 0]

        # Find unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Total number of points to sample based on the downsample_rate
        total_sample_size = int(len(points) * downsample_rate)

        # Calculate inverse frequencies
        inv_freq = 1.0 / counts
        inv_freq /= inv_freq.sum()  # Normalize to make it a probability distribution

        # Calculate initial number of samples per label before clipping
        proposed_samples = (total_sample_size * inv_freq).astype(int)

        # Calculate the minimum number of samples per label based on min_rate
        min_samples = np.ceil(total_sample_size * min_rate).astype(int)

        # Apply clipping based on the minimum rate
        num_samples_per_label = np.maximum(min_samples, proposed_samples)

        # Rebalance to fit the total sample size constraint
        if num_samples_per_label.sum() > total_sample_size:
            # Calculate excess and reduce non-clipped entries proportionally
            excess = num_samples_per_label.sum() - total_sample_size
            eligible_for_reduction = num_samples_per_label - min_samples
            reduction_fraction = excess / eligible_for_reduction.sum()
            num_samples_per_label -= (eligible_for_reduction * reduction_fraction).astype(int)

        # Initialize lists to collect sampled points and colors
        sampled_points = []
        sampled_colors = []

        # Sample points from each label group
        for label, num_samples in zip(unique_labels, num_samples_per_label):
            indices = np.where(labels == label)[0]
            sampled_indices = np.random.choice(indices, num_samples, replace=num_samples > len(indices))
            sampled_points.append(points[sampled_indices])
            sampled_colors.append(colors[sampled_indices])

        # Concatenate all sampled points and colors
        sampled_points = np.vstack(sampled_points)
        sampled_colors = np.vstack(sampled_colors)

        # Create a new Open3D point cloud with sampled data
        sampled_point_cloud = o3d.geometry.PointCloud()
        sampled_point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_point_cloud.colors = o3d.utility.Vector3dVector(sampled_colors)

        return sampled_point_cloud

    def read_point_cloud(self, filename: str):
        # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
        if ".bin" in filename:
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)
            points = np.asarray(pc_load.points, dtype=np.float64)
        else:
            sys.exit("The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)")
        preprocessed_points = self.preprocess_kitti(points, self.config.min_z, self.config.min_range)
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(
            preprocessed_points
        )  # Vector3dVector is faster for np.float64
        return pc_out

    def read_semantic_point_label(self, bin_filename: str, label_filename: str, with_labels=False):
        # read point cloud (kitti *.bin format)
        if ".bin" in bin_filename:
            if self.itlp_campus:
                points = np.fromfile(bin_filename, dtype=np.float32).reshape((-1, 3)).astype(np.float64)
            else:
                points = (
                    np.fromfile(bin_filename, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
                )
        else:
            sys.exit("The format of the imported point cloud is wrong (support only *bin)")

        # read point cloud labels (*.label format)
        if ".label" in label_filename:
            if Path(label_filename).is_file():
                if self.kitti360:
                    labels = np.fromfile(label_filename, dtype=np.int16).reshape((-1))
                else:
                    labels = np.fromfile(label_filename, dtype=np.uint32).reshape((-1))
            elif with_labels:
                return None, None
            else:
                return None
        else:
            sys.exit("The format of the imported point labels is wrong (support only *label)")

        points, sem_labels = self.preprocess_sem_kitti(
            points,
            labels,
            self.config.min_z,
            self.config.min_range,
            filter_moving=self.config.filter_moving_object,
        )
        return_labels = sem_labels

        sem_labels = (
            (np.asarray(sem_labels, dtype=np.float64) / 255.0).reshape((-1, 1)).repeat(3, axis=1)
        )  # label

        # TODO: better to use o3d.t.geometry.PointCloud(device)
        # a bit too cubersome
        # then you can use sdf_map_pc.point['positions'], sdf_map_pc.point['intensities'], sdf_map_pc.point['labels']
        pc_out = o3d.geometry.PointCloud()
        pc_out.points = o3d.utility.Vector3dVector(points)  # Vector3dVector is faster for np.float64
        pc_out.colors = o3d.utility.Vector3dVector(sem_labels)

        if with_labels:
            return pc_out, return_labels
        else:
            return pc_out

    def preprocess_kitti(self, points, z_th=-3.0, min_range=2.5):
        # filter the outliers
        z = points[:, 2]
        points = points[z > z_th]
        points = points[np.linalg.norm(points, axis=1) >= min_range]
        return points

    def preprocess_sem_kitti(self, points, labels, min_range=2.75, filter_outlier=True, filter_moving=True):
        # TODO: speed up
        if not self.itlp_campus and not self.kitti360:
            sem_labels = np.array(labels & 0xFFFF)

            range_filtered_idx = np.linalg.norm(points, axis=1) >= min_range
            points = points[range_filtered_idx]
            sem_labels = sem_labels[range_filtered_idx]

            # filter the outliers according to semantic labels
            if filter_moving:
                filtered_idx = sem_labels < 100
                points = points[filtered_idx]
                sem_labels = sem_labels[filtered_idx]

            if filter_outlier:
                filtered_idx = sem_labels != 1  # not outlier
                points = points[filtered_idx]
                sem_labels = sem_labels[filtered_idx]

            sem_labels_main_class = np.array(
                [sem_kitti_learning_map[sem_label] for sem_label in sem_labels]
            )  # get the reduced label [0-20]

        elif self.itlp_campus:
            range_filtered_idx = np.linalg.norm(points, axis=1) >= min_range
            points = points[range_filtered_idx]
            labels = labels[range_filtered_idx]
            if not self.use_outliers:
                filtered_idx = np.logical_and(
                    np.logical_and(labels != 0, labels != 5), labels < 100
                )  # not outlier
            else:
                filtered_idx = labels == 0
            sem_labels = labels[filtered_idx]
            points = points[filtered_idx]

            sem_labels_main_class = sem_labels

        elif self.kitti360:
            # print(self.kt360_id2label)
            sem_labels = np.array([self.kt360_id2kittiId[label] for label in labels])
            filtered_idx = sem_labels > 0
            points = points[filtered_idx]
            sem_labels = sem_labels[filtered_idx]

            sem_labels_main_class = sem_labels

        else:
            raise ValueError("Error: Unknown dataset")

        return points, sem_labels_main_class

    def write_merged_pc(self, out_path):
        map_down_pc_out = copy.deepcopy(self.map_down_pc)
        map_down_pc_out.transform(
            inv(self.begin_pose_inv)
        )  # back to world coordinate (if taking the first frame as reference)
        o3d.io.write_point_cloud(out_path, map_down_pc_out)
        print("save the merged point cloud map to %s\n" % (out_path))

    def __len__(self) -> int:
        if self.config.ray_loss:
            return self.ray_depth_pool.shape[0]  # ray count
        else:
            return self.sdf_label_pool.shape[0]  # point sample count

    # deprecated
    def __getitem__(self, index: int):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            sample_index = torch.range(0, self.ray_sample_count - 1, dtype=int)
            sample_index += index * self.ray_sample_count

            coord = self.coord_pool[sample_index, :]
            # sdf_label = self.sdf_label_pool[sample_index]
            # normal_label = self.normal_label_pool[sample_index]
            # sem_label = self.sem_label_pool[sample_index]
            sample_depth = self.sample_depth_pool[sample_index]
            ray_depth = self.ray_depth_pool[index]

            return coord, sample_depth, ray_depth

        else:  # use point sample
            coord = self.coord_pool[index, :]
            sdf_label = self.sdf_label_pool[index]
            # normal_label = self.normal_label_pool[index]
            # sem_label = self.sem_label_pool[index]
            weight = self.weight_pool[index]

            return coord, sdf_label, weight

    def get_batch(self):
        # use ray sample (each sample containing all the sample points on the ray)
        if self.config.ray_loss:
            train_ray_count = self.ray_depth_pool.shape[0]
            ray_index = torch.randint(0, train_ray_count, (self.config.bs,), device=self.pool_device)

            ray_index_repeat = (ray_index * self.ray_sample_count).repeat(self.ray_sample_count, 1)
            sample_index = ray_index_repeat + torch.arange(
                0, self.ray_sample_count, dtype=int, device=self.device
            ).reshape(-1, 1)
            index = sample_index.transpose(0, 1).reshape(-1)

            coord = self.coord_pool[index, :].to(self.device)
            weight = self.weight_pool[index].to(self.device)
            sample_depth = self.sample_depth_pool[index].to(self.device)

            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :].to(self.device)
            else:
                normal_label = None

            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[ray_index * self.ray_sample_count].to(
                    self.device
                )  # one semantic label for one ray
            else:
                sem_label = None

            ray_depth = self.ray_depth_pool[ray_index].to(self.device)

            return coord, sample_depth, ray_depth, normal_label, sem_label, weight

        else:  # use point sample
            train_sample_count = self.sdf_label_pool.shape[0]
            index = torch.randint(0, train_sample_count, (self.config.bs,), device=self.pool_device)
            coord = self.coord_pool[index, :].to(self.device)
            sdf_label = self.sdf_label_pool[index].to(self.device)
            origin = self.origin_pool[index].to(self.device)
            ts = self.time_pool[index].to(self.device)  # frame number or the timestamp

            if self.normal_label_pool is not None:
                normal_label = self.normal_label_pool[index, :].to(self.device)
            else:
                normal_label = None

            if self.sem_label_pool is not None:
                sem_label = self.sem_label_pool[index].to(self.device)
            else:
                sem_label = None

            weight = self.weight_pool[index].to(self.device)

            return coord, sdf_label, origin, ts, normal_label, sem_label, weight

    def add_distance_noise(self, frame_pc, base_std_dev=0.01, distance_factor=0.001):
        """Add distance-dependent Gaussian noise to an Open3D point cloud.

        :param frame_pc: Open3D point cloud object.
        :param base_std_dev: Base standard deviation of the Gaussian noise.
        :param distance_factor: Factor to adjust noise level by distance.
        :return: Noisy Open3D point cloud.
        """
        # Convert Open3D point cloud to numpy array
        points = np.asarray(frame_pc.points)

        # Calculate distances from the origin (0, 0, 0)
        distances = np.linalg.norm(points, axis=1)

        # Calculate noise standard deviation for each point
        noise_std_dev = base_std_dev + distance_factor * distances

        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std_dev[:, np.newaxis], points.shape)

        # Add noise to the original points
        noisy_points = points + noise

        # Create a new Open3D point cloud for the noisy points
        noisy_pcd = o3d.geometry.PointCloud()
        noisy_pcd.points = o3d.utility.Vector3dVector(noisy_points)

        # If the original point cloud has colors or normals, you might want to copy them to the noisy point cloud
        if frame_pc.has_colors():
            noisy_pcd.colors = frame_pc.colors
        if frame_pc.has_normals():
            noisy_pcd.normals = frame_pc.normals

        return noisy_pcd
