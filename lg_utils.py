import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
# import tensorflow as tf
import json
from typing import Dict
from glob import glob 
# from constants import DATA_DIR_PATH, FEATURE_DESCRIPTOR, FEATURE2DIM
import cv2
from tqdm import tqdm
from visualize_episodes import save_videos

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, demo_dirs, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids # index of demonstratcion which will included
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.demo_dirs = demo_dirs
        self.__getitem__(0) # initialize self.is_sim
        # print(f'Found {len(self.demo_dir)} demos')

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = True # hardcode

        episode_id = self.episode_ids[index]
        # dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        dataset_path = self.demo_dirs[episode_id]
        min_length = self.norm_stats['min_length']

        with h5py.File(dataset_path, 'r') as root:
            # is_sim = root.attrs['sim']

            ### cut min_length
            original_action_shape = root['/action/joint_pos'].shape
            trj_len = original_action_shape[0]
            start = np.random.choice((0, trj_len - min_length))
            end = start + min_length
            ###
            
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(min_length)
            # get observation at start_ts only
            qpos = root['/observation/joint_pos'][start + start_ts]
            # qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observation/{cam_name}'][start + start_ts]
            # get all actions after and including start_ts
            is_sim = False
            if is_sim:
                action = root['/action/joint_pos'][start + start_ts:end]
                action_len = min_length - start_ts
            else:
                action = root['/action/joint_pos'][max(start, start + start_ts - 1):end] # hack, to make timesteps more aligned
                action_len = min_length - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros((min_length, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(min_length)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).to(torch.float32)
        action_data = torch.from_numpy(padded_action).to(torch.float32)
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # print(image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
        return image_data.to(torch.float32), qpos_data.to(torch.float32), action_data.to(torch.float32), is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    min_length = 1000000000
    demo_dir = glob(f'{dataset_dir}/*.hdf5')
    for dataset_path in demo_dir:
        # dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observation/joint_pos'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action/joint_pos'][()]
            if len(qpos) < min_length:
                min_length = len(qpos)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos, "min_length":min_length}
    print("min length is", min_length)
    return stats

def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.99
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    print('Getting Norm Stats')
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    demo_dirs = glob(f'{dataset_dir}/*.hdf5')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(demo_dirs, train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(demo_dirs, val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
    
import os

def rename_hdf5_files(folder_path):
    file_list = os.listdir(folder_path)
    hdf5_files = [file for file in file_list if file.endswith('.hdf5')]

    # 파일 이름에서 숫자 부분 추출
    # numbers = [int(file.split('_')[1].split('.')[0]) for file in hdf5_files]
    # numbers = len(hdf5_files)

    # 빈 숫자 채우기
    # max_number = max(numbers)
    # ne# w_numbers = list(range(numbers))

    # 새로운 파일 이름 생성 및 파일 이름 변경
    for i, file in enumerate(hdf5_files):
        old_name = os.path.join(folder_path, file)
        new_name = os.path.join(folder_path, file[4:])
        print(old_name, new_name)
        os.rename(old_name, new_name) 

def parse_data_robomimic(dataset_dir, save_dir, abs_control=False):
    import robomimic
    import robomimic.utils.file_utils as FileUtils
    # the dataset registry can be found at robomimic/__init__.py
    from robomimic import DATASET_REGISTRY

    # obtain train test split
    train_ratio = 0.9

    dataset_path = dataset_dir
    f = h5py.File(dataset_path, "r")

    # each demonstration is a group under "data"
    demos = list(f["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    print("hdf5 file {} has {} demonstrations".format(dataset_path, num_demos))
    for ep in range(num_demos):
        demo_key = demos[ep]
        demo_grp = f["data/{}".format(demo_key)]
        # print(list(demo_grp['next_obs'].keys()))
        if abs_control:
            # print(demo_grp["next_obs/robot0_joint_pos"].shape, demo_grp["next_obs/robot0_gripper_qpos"].shape, demo_grp["next_obs/robot0_eef_pos"][:, -1:].shape)
            qpos_0 = np.concatenate((demo_grp["obs/robot0_joint_pos"], demo_grp["obs/robot0_gripper_qpos"][:, -1:]), axis=1)
            qpos_1 = np.concatenate((demo_grp["obs/robot1_joint_pos"], demo_grp["obs/robot1_gripper_qpos"][:, -1:]), axis=1)
            qpos = np.concatenate((qpos_0, qpos_1), axis=1)
            robot0_action = np.concatenate((demo_grp["next_obs/robot0_joint_pos"], demo_grp["next_obs/robot0_gripper_qpos"][:, -1:]), axis=1)
            robot1_action = np.concatenate((demo_grp["next_obs/robot1_joint_pos"], demo_grp["next_obs/robot1_gripper_qpos"][:, -1:]), axis=1)
            action = np.concatenate((robot0_action, robot1_action), axis=1)
            # print(demo_grp["next_obs/robot0_gripper_qpos"][0, -1], demo_grp["next_obs/robot0_gripper_qpos"][0, -2])
            # print(robot0_action.shape)
        else:
            qpos = np.concatenate((demo_grp["obs/robot0_joint_pos"], demo_grp["obs/robot1_joint_pos"]), axis=1)
            action = demo_grp["actions"]
        image = demo_grp["obs/agentview_image"]

        max_timesteps = len(image)

        data_dict = {
            '/observations/qpos': qpos,
            '/action': action,
            '/observations/images/camera': image,
        }

        with h5py.File(f'{save_dir}/episode_{ep}.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            image.create_dataset('camera', (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3), )
            if not abs_control:
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))
            else:
                qpos = obs.create_dataset('qpos', (max_timesteps, 16))
                action = root.create_dataset('action', (max_timesteps, 16))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'{save_dir}/episode_{ep}.hdf5 - saved')

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)