import csv

import numpy as np
from numpy.linalg import inv
from pyquaternion import Quaternion


def read_calib_file(filename):
    """Read calibration file (with the kitti format)
    returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)
    key_num = 0

    for line in calib_file:
        # print(line)
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib


def remap_poses(poses, new_origin):
    # Ensure the new_origin_id is within the valid range

    # Extract the new origin pose
    new_origin_pose = new_origin

    # Compute the inverse of the new origin pose
    new_origin_inv = np.linalg.inv(new_origin_pose)

    # Transform all poses with respect to the new origin
    remapped_poses = []
    for pose in poses:
        if pose is None:
            remapped_poses.append(None)
            continue
        pose = np.array(pose)
        new_pose = new_origin_inv @ pose
        remapped_poses.append(new_pose.tolist())

    return remapped_poses


def read_kitti360_poses_file(filename, calibration):
    """Read pose file (with the kitti format)"""
    pose_file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    print(f"Tr:\n{Tr}")
    Tr_inv = inv(Tr)
    # invert_z_axis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    pose_id = 1
    for line in pose_file:
        values = [float(v) for v in line.strip().split()]
        cur_id = int(values[0])
        while pose_id < cur_id:
            pose_id += 1
            poses.append(None)

        values = values[1:]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        # poses.append(pose)
        # print(poses[-1])
        # input()
        poses.append(pose)
        pose_id += 1

    origin_translation = np.zeros((4, 4))
    origin_translation[:3, 3] = poses[0][:3, 3]

    translated_poses = []
    for pose in poses:
        if pose is None:
            translated_poses.append(None)
            continue
        p = pose - origin_translation
        # translated_poses.append(np.matmul(Tr_inv, np.matmul(p, Tr)))  # lidar pose in world frame
        translated_poses.append(Tr_inv @ p @ Tr)  # lidar pose in world frame

        # print(translated_poses[-1])
        # input()

    pose_file.close()
    # print(f"Read {translated_poses[:5]} poses")
    poses = remap_poses(translated_poses, translated_poses[0])
    return poses


def read_poses_file(filename, calibration):
    """Read pose file (with the kitti format)"""
    pose_file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))  # lidar pose in world frame

    pose_file.close()
    return poses


def csv_odom_to_transforms(path):
    # odom_tfs = {}
    poses = []
    with open(path, mode="r") as f:
        reader = csv.reader(f)
        # get header and change timestamp label name
        header = next(reader)
        header[0] = "ts"
        # Convert string odometry to numpy transfor matrices
        for row in reader:
            odom = {l: row[i] for i, l in enumerate(header)}
            # Translarion and rotation quaternion as numpy arrays
            trans = np.array([float(odom[l]) for l in ["tx", "ty", "tz"]])
            quat = Quaternion(np.array([float(odom[l]) for l in ["qx", "qy", "qz", "qw"]]))
            rot = quat.rotation_matrix
            # Build numpy transform matrix
            odom_tf = np.eye(4)
            odom_tf[0:3, 3] = trans
            odom_tf[0:3, 0:3] = rot
            # Add transform to timestamp indexed dictionary
            # odom_tfs[odom["ts"]] = odom_tf
            poses.append(odom_tf)

    return poses
