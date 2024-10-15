import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import imageio
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.env_utils as EnvUtils
from collections import namedtuple


State = namedtuple("State", "observation reward")

class RobomimicWrapper:
    def __init__(self, env, abs_control=False):
        self.env = env
        self.max_timesteps = 1000
        self.max_reward = 1
        self.abs_control = abs_control
        if abs_control:
            self.cstate = None

    def set_cstate(self, qpos):
        self.cstate = qpos
        # self.cstate[7] = 0
        # self.cstate[-1] = 0

    def reset(self):
        state = self.env.reset()
        if self.abs_control:
            qpos_0 = np.concatenate((state["robot0_joint_pos"], state["robot0_gripper_qpos"][-1:]), axis=0)
            qpos_1 = np.concatenate((state["robot1_joint_pos"], state["robot1_gripper_qpos"][-1:]), axis=0)
            qpos = np.concatenate((qpos_0, qpos_1), axis=0)
        else:
            qpos = np.concatenate((state['robot0_joint_pos'], state['robot1_joint_pos']))
        obs = {
            'images': {
                'camera' : state['agentview_image']
            },
            'qpos':qpos
        }
        output = State(obs, 0)

        if self.abs_control:
            self.set_cstate(qpos)

        return output

    def adjust_gripper(self, gs):
        if np.abs(gs) < 0.005:
            return 0
        else:
            return gs

    def step(self, action):
        if self.abs_control:
            action = action - self.cstate
            action[7] = self.adjust_gripper(action[7])
            action[-1] = self.adjust_gripper(action[-1])
            # print('after', action[7], action[-1])
        state, reward, done, _ = self.env.step(action)
        if self.abs_control:
            qpos_0 = np.concatenate((state["robot0_joint_pos"], state["robot0_gripper_qpos"][-1:]), axis=0)
            qpos_1 = np.concatenate((state["robot1_joint_pos"], state["robot1_gripper_qpos"][-1:]), axis=0)
            qpos = np.concatenate((qpos_0, qpos_1), axis=0)
        else:
            qpos = np.concatenate((state['robot0_joint_pos'], state['robot1_joint_pos']))
        obs = {
            'images': {
                'camera' : state['agentview_image']
            },
            'qpos':qpos
        }
        output = State(obs, reward)

        if self.abs_control:
            self.set_cstate(qpos)
        return output

def get_robomimic_env(task_name):
    if task_name[-3:] == 'abs':
        abs_control = True
        task_name = task_name[:-4]
    else:
        abs_control = False
    dataset_dir = f'/home/jellyho/robomimic/datasets/{task_name}/ph/demo_v141.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_dir)
    print(env_meta)
    if abs_control:
        env_meta['env_kwargs']['controller_configs']['type'] = 'JOINT_POSITION'
        env_meta['env_kwargs']['controller_configs']['output_min'] = [-1] * 7
        env_meta['env_kwargs']['controller_configs']['output_max'] = [1] * 7
        env_meta['env_kwargs']['controller_configs']['interpolation'] = 'linear'
        env_meta['env_kwargs']['controller_configs']['kp'] = 150
        env_meta['env_kwargs']['controller_configs']['damping'] = 1
        env_meta['env_kwargs']['reward_shaping'] = True
        # env_meta['env_kwargs']['controller_configs']['kp_limits'] = (0, 300)
        # env_meta['env_kwargs']['controller_configs']['damping_ratio_limits'] = (0, 100)

    env_meta['env_kwargs']['camera_heights'] = 240
    env_meta['env_kwargs']['camera_widths'] = 360

    print(env_meta)

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=['agentview'], 
        camera_height=240, 
        camera_width=360,
        reward_shaping=False
    )


    return RobomimicWrapper(env, abs_control)

if __name__ == "__main__": 
    env = get_robomimic_env('transport_abs')
    ts = env.reset()
    image_list = []
    # print(env.env.action_dim)
    # create a video writer
    import h5py
    # for i in range(200):
        # ts = env.reset()
    f = h5py.File(f'../datasets/transport_abs/episode_1.hdf5', 'r')
    action = f['/action'][()]
    for a in action:
        # print(a)
        ts = env.step(a)
        # ts = env.step(np.random.randn(16,))
        # print(ts)
        obs = ts.observation
        if 'images' in obs:
            image_list.append(obs['images'])
        else:
            image_list.append({'main': obs['image']})
    from visualize_episodes import save_videos
    save_videos(image_list, 0.02, video_path='test.mp4')

    # print(env.step(np.random.randn(16,)))
